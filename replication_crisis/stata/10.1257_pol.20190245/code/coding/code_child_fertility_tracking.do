*** 1. INTRODUCTION ***
/*
Description: 	Tracks attrition and fertility measures of children who were originally age 6-15 in baseline
Uses: 			Raw survey data
Creates: 		tracking_child6to15_attrition_for_fertility.dta
*/

*** 2. SET ENVIRONMENT ***

version 14.1 		
set more off		
clear all			
pause on 			


//Prompt for directory if not filled in above
if "$PKH" == "" {
	display "Enter the directory where the PKH files can be located: " _request(PKH)
	* add a / if they forgot one
	quietly if substr("$PKH",length("PKH"),1) != "/" & substr("$PKH",length("$PKH"),1) != "\" {
		global PKH = "$PKH/"	
	}
}

local PKH = "$PKH"
local pkhdata = "`PKH'data/raw/"
local output = "`PKH'output"
local TARGET2 = "$TARGET2"

//Set working directory:
cd "`pkhdata'"

*** 3. APPENDING WAVE III AND IV OBSERVATIONS & SCHOOL ATTENDANCE DATA ***
//load Wave IV survey date information
use "Wave IV/A_KJ.dta", clear

//destring interview date
destring aivw_dd aivw_mm aivw_yy, replace

//drop duplicates and improperly coded observations 
duplicates report aid 
duplicates drop aid, force
drop if (aivw_yy < 2009 | aivw_mm > 12 | aivw_dd > 31) 

tempfile w4_dates
save `w4_dates', replace 

//load Wave IV attendance information
use "Wave IV/A_DLA1TYPE.dta", clear

//Drop irrelevant variables and missing values of adla1type 
keep aid adla1type adla17 adla18
drop if missing(adla1type) 

//Recode values for 'No' entries
replace adla17=0 if adla17==3
replace adla18=0 if adla18==3

//Condense to wide format
reshape wide adla17 adla18, i(aid) j(adla1type)

//Generate the total number of days school open in the last week
egen days_open_lastweek = rowtotal(adla17*), missing

//Generate the total number of days school attended in the last week
egen days_attend_lastweek = rowtotal(adla18*), missing
replace days_attend_lastweek=. if days_open_lastweek==0

sort aid 

tempfile w4_attendance
save `w4_attendance', replace 



//Load Wave IV child data and prepare for append
use "Wave IV/A.dta", clear

//drop duplicates
duplicates report aid 
duplicates drop aid, force
sort aid

//merge survey date info
merge 1:1 aid using `w4_dates'
tab _merge
rename _merge _merge_w4

//merge attendance info 
merge 1:1 aid using `w4_attendance'
tab _merge 
rename _merge _merge_attendance_w4


//destring other schooling variables 
replace adla10 = "" if adla10 == "TT" | adla10 == "EX"
destring adla10, replace

//drop new observations with anomalous household ID values 
gen drop_indicator = substr(aid,12,2)
drop if drop_indicator == "13"
drop drop_indicator 


//generate id variable for use in fertility merge 
//need to do this in Wave IV only because of coding particularity in how aid gets assigned
tostring air01_cd, replace format(%02.0f)
gen aid_original = substr(aid, 1, 17) + air01_cd

//generate 15-digit aid variable for use in appending with Wave III 
rename aid aid_w4
gen aid = substr(aid_w4, 1, 7) + substr(aid_w4, 10, 6) + substr(aid_w4, 18, 2)


//survey round 
gen survey_round = 2

tempfile child_w4
save `child_w4', replace



//first load Wave III dataset containing survey date information
use "Wave III/Data/A_KJ_fin.dta", clear

//drop duplicates and improperly coded observations 
duplicates drop aid, force
drop if (aivw_yy < 2009 | aivw_mm > 12 | aivw_dd > 31)

tempfile w3_dates
save `w3_dates', replace 

//load Wave III attendance information
use "Wave III/Data/A_DLA1TYPE_fin.dta", clear

//Drop irrelevant variables
keep aid adla1type adla17 adla18

//Recode values for 'No' entries
replace adla17=0 if adla17==3
replace adla18=0 if adla18==3

//Condense to wide format
reshape wide adla17 adla18, i(aid) j(adla1type)

//Generate the total number of days school open in the last week
egen days_open_lastweek = rowtotal(adla17*), missing

//Generate the total number of days school attended in the last week
egen days_attend_lastweek = rowtotal(adla18*), missing
replace days_attend_lastweek=. if days_open_lastweek==0

sort aid 

tempfile w3_attendance
save `w3_attendance', replace 




//Load Wave III household roster to obtain previous ID numbers where applicable 
use "Wave III/Data/R_AR_01_fin.dta", clear
duplicates drop 

//convert current/previous id numbers to string
tostring rar00_07 rar00, replace format(%02.0f)

//generate aid number from roster for merging  
gen aid = substr(rid,1,3) + "3" + substr(rid,5,9) + rar00
gen hh_split_indicator = substr(rid,12,2)

//keep only relevant variables 
keep aid rar00 rar00_07 hh_split_indicator

tempfile roster_w3
save `roster_w3', replace 

//Load Wave III child data and drop (1) duplicate
use "Wave III/Data/A_fin.dta", clear

gen aid_original = aid

//drop duplicates
	/*this step is before the creation of 9-digit aid because duplicate 9-digit
	aids can refer to different unique children*/
duplicates report aid 
duplicates drop aid, force
sort aid

//merge survey date info
merge 1:1 aid using `w3_dates'
tab _merge 
rename _merge _merge_w3

//merge attendance info 
merge 1:1 aid using `w3_attendance'
tab _merge 
rename _merge _merge_attendance_w3

//gen survey round variable 
gen survey_round = 1

//append Wave IV 
append using `child_w4', force 
sort aid survey_round 


//we want aid and survey_round to uniquely identify kids for the purpose of matching back to Wave I
//drop kids who were new in Wave IV 
drop if survey_round == 2 & acov3 == 3
//drop one extra anomalous observation
duplicates drop aid survey_round, force


//merge roster info, will be incorporated into Wave I aid number 
merge m:1 aid using `roster_w3'
tab _merge 
//only keep roster info for people who are actually aged 6-15 
drop if _merge == 2
drop _merge 

//need to drop if split household and rar00_07 == 0 (means was not in baseline survey so can be disregarded)
drop if hh_split_indicator != "00" & rar00_07 == "00"

//create 11-digit (Wave I) aid
rename aid aid_w3
gen aid = substr(aid_w3, 1, 9) + substr(aid_w3, 14, 2)
//use old (Wave I) household serial number if different from current one 
replace aid = substr(aid_w3, 1, 9) + rar00_07 if rar00_07 != "00" & !missing(rar00_07)

tempfile child_w3_w4
save `child_w3_w4', replace



//load Wave I attendance information
use "Wave I/Data/A_DLA1TYPE_fin.dta", clear

//Drop irrelevant variables
keep aid adla1type adla17 adla18

//Recode values for 'No' entries
replace adla17=0 if adla17==3
replace adla18=0 if adla18==3

//Condense to wide format
reshape wide adla17 adla18, i(aid) j(adla1type)

//Generate the total number of days school open in the last week
egen days_open_lastweek = rowtotal(adla17*), missing

//Generate the total number of days school attended in the last week
egen days_attend_lastweek = rowtotal(adla18*), missing
replace days_attend_lastweek=. if days_open_lastweek==0

sort aid 

tempfile w1_attendance
save `w1_attendance', replace 

//Load Wave I as master dataset
use "Wave I/Data/A_fin.dta", clear

//merge Wave I attendance data  
merge 1:1 aid using `w1_attendance'
tab _merge 
rename _merge _merge_attendance_w1

//rename adla22a_n variable to match name in waves III and IV
rename adla22a_n adla22_an  

//generate survey round variable 
gen survey_round = 0

//Append Wave III and Wave IV
//generate variable, "survey_round", to identify wave of observations 
//(0 = Wave I, 1 = Wave 2, 2 = Wave III)
append using `child_w3_w4', force
sort aid survey_round

tempfile child_allwaves
save `child_allwaves', replace


*** 5. CODING EDUCATION OUTCOME VARIABLES ***
//Most code adapted from Generasi evaluation do files


//Generate survey dates using first survey attempt with at least partial completion
//different variable names for Wave I data 
gen survey_date=mdy(aivw_mm_1, aivw_dd_1, aivw_yy_1) if aahkw_1==1 | aahkw_1==2 & survey_round == 0
replace survey_date=mdy(aivw_mm_2, aivw_dd_2, aivw_yy_2) if survey_date==. & (aahkw_2==1 | aahkw_2==2) & survey_round == 0
replace survey_date=mdy(aivw_mm_3, aivw_dd_3, aivw_yy_3) if survey_date==. & (aahkw_3==1 | aahkw_3==2) & survey_round == 0
//Waves III and IV 
replace survey_date = mdy(aivw_mm, aivw_dd, aivw_yy) if (survey_round == 1 | survey_round == 2) 

//account for 127 Wave III observations that did not have survey date information
gen survey_start_day = substr(astart_d,1,2) if _merge_w3 == 1
destring survey_start_day, replace 
gen survey_start_mon = substr(astart_d,4,2) if _merge_w3 == 1
destring survey_start_mon, replace 
gen survey_start_year = substr(astart_d,7,2) if _merge_w3 == 1
destring survey_start_year, replace 
replace survey_start_year = survey_start_year + 2000 if _merge_w3 == 1
replace survey_date = mdy(survey_start_mon, survey_start_day, survey_start_year) if _merge_w3 == 1
drop survey_start*


***** 7. TRACKING ATTRITION OF CHILDREN FROM BASELINE SURVEY *****
//drop/do not consider respondents that were added in later waves 
bysort aid (survey_round): gen delete = 1 if acov3[1] == 3 & survey_round[1] != 0
drop if delete == 1
drop delete 


bysort aid (survey_round): gen delete = 1 if survey_round[1] != 0
drop if delete == 1
drop delete 

//identify baseline observations that were not tracked down 
preserve 
duplicates tag aid, gen(dup)
expand 2 if dup==0, gen(temp1)
expand 2 if dup==0 & temp1==1, gen(temp2)
replace survey_round = 1 if temp1 == 1 & temp2 == 0
replace survey_round = 2 if temp2 == 1
gen not_surveyed = (temp1 == 1 | temp2 == 1)


expand 2 if dup==1 & survey_round == 2, gen(temp3)
replace survey_round = 1 if temp3 == 1
replace not_surveyed = 1 if temp3 == 1

expand 2 if dup==1 & survey_round == 1 & temp3 != 1, gen(temp4)
replace survey_round = 2 if temp4 == 1
replace not_surveyed = 1 if temp4 == 1

sort aid survey_round 
keep aid survey_round not_surveyed

//keep only non-surveyed observations
keep if not_surveyed == 1

tempfile attrition
save `attrition', replace 
restore 

gen not_surveyed = 0
append using `attrition'
sort aid survey_round 

*** 8. MERGING HOUSEHOLD-LEVEL COVARIATES ***

//create rid variable to match children with households from HH file
gen rid = substr(aid, 1, 3) + "1" + substr(aid, 5, 5)

//create baseline marker for merging with household covariates 
gen merge_survey_round = 0

tempfile child_allwaves_coded
save `child_allwaves_coded', replace 

//load master household dataset 
cd "`PKH'"
use "data/coded/household_allwaves_master.dta", clear 

//keep only relevant household covariates 
keep rid survey_round L07 *baseline_nm *miss ///
	 kabupaten kecamatan province 

//create corresponding marker for merging with child observations 
gen merge_survey_round = survey_round 
//keep only baseline households for this part 
keep if merge_survey_round == 0

tempfile hh_covariates
save `hh_covariates', replace

//merge 
use `child_allwaves_coded', clear 
merge m:1 rid merge_survey_round using `hh_covariates'
tab _merge
keep if _merge == 3
drop _merge merge_survey_round

tempfile child_allwaves_coded_withcov
save `child_allwaves_coded_withcov', replace 

//now just need to merge PKH receipt status 
//load master household dataset 
cd "`PKH'"
use "data/coded/household_allwaves_master.dta", clear 

//keep only necessary variables for merge 
keep rid* survey_round received* pkh* non_pkh_hh_cct_kec split_indicator

tempfile hh_treatment
save `hh_treatment', replace 

//merge 
use `child_allwaves_coded_withcov', clear 


//generate rid_merge variable to ensure merge to correct household (accounts for HH splits after baseline)
gen rid_merge = substr(aid, 1, 3) + "1" + substr(aid,5,5) if survey_round == 0
replace rid_merge = substr(aid_original, 1, 3) + "1" + substr(aid_original,5,9) if survey_round == 1
replace rid_merge = substr(aid_original, 1, 3) + "1" + substr(aid_original,5,13) if survey_round == 2 

merge m:1 rid_merge survey_round using `hh_treatment'
tab _merge 

//still want to keep people who don't merge to a household (this is from HH attrition)
drop if _merge == 2
drop _merge 

//generate baseline age variable for attrition-by-age regression
gen age_years_baseline = age_years if survey_round == 0
bysort aid (survey_round): carryforward age_years_baseline, replace

//generate baseline gender variabel for attrition-by-gender regression
gen gender_baseline = air02 if survey_round == 0
bysort aid (survey_round): carryforward gender_baseline, replace 
//recode girls from 3 to 0
recode gender_baseline (3 = 0)
label define gender 0 "Female" 1 "Male"
label values gender_baseline gender 




*** 9. INVESTIGATE CHILDREN WHO MOVED OUT OF THE HOUSEHOLD ***
//need to carryforward aid_w3 values so that can merge "moved out" observations
//(because these are by construction missing for people who attrited in Wave IV!)
bysort aid (survey_round): carryforward aid_w3, replace 

//generate aid variable for merging with roster info from R_AR_18
gen aid_merge = aid 
replace aid_merge = aid_w3 if survey_round == 2 & !missing(aid_w3)

//recast for merge
recast str15 aid_merge

tempfile attrition_allwaves 
save `attrition_allwaves', replace 

//open R_AR_18 (moved out within past 12 months) roster for Wave IV 
cd "`pkhdata'"
use "Wave IV/R_AR_18.dta", clear 
duplicates drop

//drop if no "moved out" observations for a household 
drop if missing(rar18)
//convert rar18_09 variable to string (note: "rar18_09" is a mislabeling; actually links to main household roster this wave) 
tostring rar18_09, replace format(%02.0f)
//need to drop people who have serial number "00"; this comes about occasionally for people who were not in survey last wave 
drop if rar18_09 == "00"
//create id variable for merging 
gen id = rid + rar18_09

tempfile moved_w4
save `moved_w4', replace 

//open R_AR_01 (household roster) for Wave IV 
use "Wave IV/R_AR_01.dta", clear 
duplicates drop 

//convert rar00 variable to string 
tostring rar00, replace format(%02.0f)
//create id variable for merging 
gen id = rid + rar00 
//merge with roster of people who moved 
merge 1:1 id using `moved_w4'
tab _merge //all should be _merge == 1 or 3
//tag people we can identify in R_AR_18 as having moved out
gen moved_out = (_merge == 3)  
drop _merge 

//construct Wave III aid number 
tostring rar00_09, replace format(%02.0f)
//drop if serial number is "00" to match above; person was not in survey last wave 
drop if rar00_09 == "00"
gen aid_merge = substr(rid, 1, 3) + "3" + substr(rid, 5, 3) + substr(rid, 10, 6) + rar00_09

//survey round 
gen survey_round = 2

//save tempfile
tempfile moved_out_roster_w4
save `moved_out_roster_w4', replace 


//open R_AR_18 (moved out within past 12 months) roster for Wave III
use "Wave III/Data/R_AR_18_fin.dta", clear
duplicates drop 
//convert rar18_07 variable to string (note: "rar18_07" is a mislabeling; actually links to main household roster this wave) 
tostring rar18_07, replace format(%02.0f)
//need to drop people who have serial number "00"
drop if rar18_07 == "00"
//create id variable for merging 
gen id = rid + rar18_07

tempfile moved_w3
save `moved_w3', replace 

//open R_AR_01 (household roster) for Wave IV 
use "Wave III/Data/R_AR_01_fin.dta", clear 
duplicates drop 

//convert rar00 variable to string 
tostring rar00, replace format(%02.0f)
//create id variable for merging 
gen id = rid + rar00 
//merge with roster of people who moved 
merge 1:1 id using `moved_w3'
tab _merge //all but 1 anomalous observation should be _merge == 1 or 3
drop if _merge == 2
//tag people we can identify in R_AR_18 as having moved out
gen moved_out = (_merge == 3)  
drop _merge 

//need to drop individuals who live in split households but weren't in previous wave (leads to duplicate aid numbers)
gen split_code = substr(rid, 12, 2)
drop if split_code != "00" & rar00_07 == 0

//construct Wave III aid number 
tostring rar00_07, replace format(%02.0f)
gen aid_merge = substr(rid, 1, 3) + "3" + substr(rid, 5, 5) + rar00 
replace aid_merge = substr(rid, 1, 3) + "3" + substr(rid, 5, 5) + rar00_07 if rar00_07 != "00" & !missing(rar00_07)

//survey round 
gen survey_round = 1

//append Wave IV roster with moved-out observations 
append using `moved_out_roster_w4', force

//take care of duplicate cases 
duplicates tag aid_merge survey_round, gen(dup)
//address issue of people being reported as "moved" but surveyed elsewhere as split HH
drop if dup > 0 & rar01a == 2
drop dup 
//the remaining duplicates are either from Generasi HHs that split in Wave II (which we don't count)
//or are duplicates of the same individual. None of them should have a "moved_out" value of 1, and we can simply drop 
duplicates drop aid_merge survey_round, force 


//save tempfile 
tempfile moved_out_roster_all
save `moved_out_roster_all', replace 

//reopen main attrition file 
use `attrition_allwaves', clear 
//merge with roster info 
merge 1:1 aid_merge survey_round using `moved_out_roster_all'
//drop observations that only come from roster (i.e., they are not children we are tracking)
drop if _merge == 2
//observations for which _merge == 1 are either baseline observations or have attritted (not_surveyed == 1)!
//generate indicator for individual being surveyed in HH roster 
gen in_roster = (_merge == 3)
replace in_roster = 1 if survey_round == 0

gen still_in_hh = (not_surveyed == 1 & in_roster == 1 & moved_out != 1)
gen migrated = (not_surveyed == 1 & moved_out == 1)
drop _merge 

tempfile attrition_with_roster_info
save `attrition_with_roster_info', replace 

//now want to account for children whose households attrited 
cd "`PKH'data/coded"
use "household_allwaves_master", clear 

//drop all but rid and survey_round info 
keep rid survey_round
duplicates drop rid survey_round, force

tempfile household_merge
save `household_merge', replace 

//re-open attrition panel 
use `attrition_with_roster_info', clear 

//merge with household info 
merge m:1 rid survey_round using `household_merge' 
tab _merge 
drop if _merge == 2

//generate indicator for household attrition 
gen household_attrited = (_merge == 1)
drop _merge 
bysort aid (survey_round): gen household_attrited_midline = (household_attrited[2] == 1)
bysort aid (survey_round): gen household_attrited_endline = (household_attrited[3] == 1)
gen household_ever_attrited = household_attrited
replace household_ever_attrited = . if survey_round == 2 & household_ever_attrited == 0
bysort aid (survey_round): carryforward household_ever_attrited, replace 

//carry forward relevant variables 
gen ever_migrated = migrated
replace ever_migrated = . if survey_round == 2 & ever_migrated == 0
bysort aid (survey_round): carryforward ever_migrated, replace


//variables for tabulation 
//baseline indicator (1 for everyone)
gen child_in_baseline = 1
//should be no overlap between these two
bysort aid (survey_round): gen migrated_midline = (migrated[2] == 1)
bysort aid (survey_round): gen migrated_endline = (migrated[3] == 1)  
//moved but not tracked in R_AR_18
gen moved_away = (still_in_hh == 1 & rar01a == 2)
bysort aid (survey_round): gen moved_away_midline = (moved_away[2] == 1)
bysort aid (survey_round): gen moved_away_endline = (moved_away[3] == 1)
gen ever_moved_away = moved_away
replace ever_moved_away = . if survey_round == 2 & ever_moved_away == 0
bysort aid (survey_round): carryforward ever_moved_away, replace
//died 
gen died = (still_in_hh == 1 & rar01a == 4)
bysort aid (survey_round): gen died_midline = (died[2] == 1)
bysort aid (survey_round): gen died_endline = (died[3] == 1) 
gen ever_died = died 
replace ever_died = . if survey_round == 2 & ever_died == 0
bysort aid (survey_round): carryforward ever_died, replace 
//still in HH panel survey 
gen panel_in_roster = (rar01a == 1 & (not_surveyed == 1 | still_in_hh == 1))
bysort aid (survey_round): gen panel_in_roster_midline = (panel_in_roster[2] == 1)
bysort aid (survey_round): gen panel_in_roster_endline = (panel_in_roster[3] == 1)
gen ever_panel_in_roster = panel_in_roster
replace ever_panel_in_roster = . if survey_round == 2 & ever_panel_in_roster == 0
bysort aid (survey_round): carryforward ever_panel_in_roster, replace 

//indicator for if child is accounted for 
gen accounted_for = (household_attrited == 1 | ever_moved_away == 1 | ever_died == 1 | ever_migrated == 1 | ever_panel_in_roster == 1)
replace accounted_for = 1 if not_surveyed == 0
gen unaccounted_for = (accounted_for == 0)
gen surveyed = 1 - not_surveyed

//additional checks: prevent double-counting by taking care of cases where child moved out in midline then household attrited in endline:
replace household_attrited = 0 if (survey_round == 2 & household_attrited == 1 & (moved_away_midline == 1 | migrated_midline == 1 | died_midline == 1))
//one additional case where individual is marked as both "migrated" and "panel"
replace panel_in_roster_endline = 0 if migrated_endline == 1

//reason for migrating (for those who are in R_AR_18)
gen migrated_for_school = (rar24 == 1) if !missing(rar24) & migrated == 1
gen migrated_for_work = (rar24 == 2) if !missing(rar24) & migrated == 1
gen migrated_for_marriage = (rar24 == 3) if !missing(rar24) & migrated == 1
gen migrated_for_other = (rar24 > 3) if !missing(rar24) & migrated == 1


sort aid survey_round 
//save
compress
cd "`PKH'data/coded"
save "tracking_child6to15_attrition_for_fertility", replace


