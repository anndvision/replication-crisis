*** 1. INTRODUCTION ***
/*
Description: 	Creates panel dataset of children aged 6-15 years at time of survey
Uses: 			Raw survey data
Creates: 		child6to15_allwaves_master.dta
*/

*** 2. SET ENVIRONMENT ***

version 14.1 		
set more off		
clear all			
pause on 			

//Prompt for directory if not filled in above
if "$PKH" == "" {
	display "Enter the directory where the PKH files can be located: " _request(PKH)
	* add a \ if they forgot one
	quietly if substr("$PKH",length("PKH"),1) != "/" & substr("$PKH",length("$PKH"),1) != "\" {
		global PKH = "$PKH/"	
	}
}

local PKH = "$PKH"
local pkhdata = "`PKH'data/raw/"
local output = "`PKH'output"
local TARGET2 = "$TARGET2"

cd "`PKH'"


//Set working directory:
cd "`pkhdata'"

*** 3. APPENDING WAVE III AND IV OBSERVATIONS & SCHOOL ATTENDANCE DATA ***
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

//Load Wave III child data and drop (1) duplicate
use "Wave III/Data/A_fin.dta", clear

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

//create 11-digit (Wave I) aid
rename aid aid_w3
gen aid = substr(aid_w3, 1, 9) + substr(aid_w3, 14, 2)

tempfile child_w3
save `child_w3', replace

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

//generate 9-digit aid variable
rename aid aid_w4
gen aid = substr(aid_w4, 1, 7) + substr(aid_w4, 10, 2) + substr(aid_w4, 18, 2)

//destring birth date and serial number variables 
destring air01_cd air03_mm air03_yy, replace 

//destring other schooling variables 
replace adla10 = "" if adla10 == "TT" | adla10 == "EX"
destring adla10, replace

tempfile child_w4
save `child_w4', replace

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
use "Wave I/Data/2_A_fin.dta", clear

//merge Wave I attendance data  
merge 1:1 aid using `w1_attendance'
tab _merge 
rename _merge _merge_attendance_w1

//rename adla22a_n variable to match name in waves III and IV
rename adla22a_n adla22_an  

//Append Wave III and Wave IV
//generate variable, "survey_round", to identify wave of observations 
//(0 = Wave I, 1 = Wave 2, 2 = Wave III)
append using `child_w3' `child_w4', generate(survey_round) force
sort aid survey_round

tempfile child_allwaves
save `child_allwaves', replace

*** 4. CODING PKH RANDOMIZATION/TREATMENT STATUS ***
use "Wave I/Data/pkhben2013_use.dta", clear

//kep only ea, randomization, PKH status, province code
keep ea L07 K09 K13 prov09 

//keep one observation per ea (kecamatan)
duplicates drop ea, force

//drop extra 90 kecs in order to restrict to original 360 kecamatans
drop if missing(L07)

tempfile ea_treatment_list
save `ea_treatment_list'

//use merged all-wave household data
use `child_allwaves', clear

//generate ea string
gen ea = substr(aid, 1, 3)

//merge with kecamatan treatment list 
merge m:1 ea using `ea_treatment_list' 
tab _merge
//only keep observations from original 360 kecamatans (L07 is non-missing)
keep if _merge == 3
drop _merge
sort aid survey_round

*** 5. CODING EDUCATION OUTCOME VARIABLES ***
//Generate survey dates using first survey attempt with at least partial completion
//different variable names for Wave I data 
gen survey_date=mdy(aivw_mm_1, aivw_dd_1, aivw_yy_1) if aahkw_1==1 | aahkw_1==2 & survey_round == 0
replace survey_date=mdy(aivw_mm_2, aivw_dd_2, aivw_yy_2) if survey_date==. & (aahkw_2==1 | aahkw_2==2) & survey_round == 0
replace survey_date=mdy(aivw_mm_3, aivw_dd_3, aivw_yy_3) if survey_date==. & (aahkw_3==1 | aahkw_3==2) & survey_round == 0
//Waves III and IV 
replace survey_date = mdy(aivw_mm, aivw_dd, aivw_yy) if (survey_round == 1 | survey_round == 2) 

//account for Wave III observations that did not have survey date information
gen survey_start_day = substr(astart_d,1,2) if _merge_w3 == 1
destring survey_start_day, replace 
gen survey_start_mon = substr(astart_d,4,2) if _merge_w3 == 1
destring survey_start_mon, replace 
gen survey_start_year = substr(astart_d,7,2) if _merge_w3 == 1
destring survey_start_year, replace 
replace survey_start_year = survey_start_year + 2000 if _merge_w3 == 1
replace survey_date = mdy(survey_start_mon, survey_start_day, survey_start_year) if _merge_w3 == 1
drop survey_start*

//Generate age 7 to 12 enrollment dummy
gen enroll_age7to12 = .
replace enroll_age7to12 = 1 if (adla09 == 1 | adla09a == 1) & age_years >= 7 & age_years <= 12
replace enroll_age7to12 = 0 if adla01 == 3 & age_years >= 7 & age_years <= 12
replace enroll_age7to12 = 0 if adla04 == 96 & age_years >= 7 & age_years <= 12
replace enroll_age7to12 = 0 if adla09 == 3 & age_years >= 7 & age_years <= 12


//Generate age 7 to 12 and primary school enrollment dummy
gen enroll_age7to12_SD = .
replace enroll_age7to12_SD = 1 if (adla09 == 1 | adla09a == 1) & age_years >= 7 & age_years <= 12 & (adla04 == 1 | adla04 == 2)
replace enroll_age7to12_SD = 0 if adla01 == 3 & age_years >= 7 & age_years <= 12
replace enroll_age7to12_SD = 0 if adla04 == 96 & age_years >= 7 & age_years <= 12
replace enroll_age7to12_SD = 0 if (adla09 == 1 | adla09a == 1) & age_years >= 7 & age_years <= 12 & adla04 != 1 & adla04 != 2
replace enroll_age7to12_SD = 0 if adla09 == 3 & age_years >= 7 & age_years <= 12


//Generate age 13 to 15 enrollment dummy
gen enroll_age13to15 = .
replace enroll_age13to15 = 1 if (adla09 == 1 | adla09a == 1) & age_years >= 13 & age_years <= 15
replace enroll_age13to15 = 0 if adla01 == 3 & age_years >= 13 & age_years <= 15
replace enroll_age13to15 = 0 if adla04 == 96 & age_years >= 13 & age_years <= 15
replace enroll_age13to15 = 0 if adla09 == 3 & age_years >= 13 & age_years <= 15


//Generate age 13 to 15 and secondary school enrollment dummy
gen enroll_age13to15_SMP = .
replace enroll_age13to15_SMP = 1 if (adla09 == 1 | adla09a == 1) & age_years >= 13 & age_years <= 15 & (adla04 == 3 | adla04 == 4)
replace enroll_age13to15_SMP = 0 if adla01 == 3 & age_years >= 13 & age_years <= 15
replace enroll_age13to15_SMP = 0 if adla04 == 96 & age_years >= 13 & age_years <= 15
replace enroll_age13to15_SMP = 0 if (adla09 == 1 | adla09a == 1) & age_years >= 13 & age_years <= 15 & adla04 != 3 & adla04 != 4 
replace enroll_age13to15_SMP = 0 if adla09 == 3 & age_years >= 13 & age_years <= 15


//Generate age 13 to 15 and other (non-secondary) school enrollment dummy
gen enroll_age13to15_other = .
replace enroll_age13to15_other = 1 if (adla09 == 1 | adla09a == 1) & age_years >= 13 & age_years <= 15 & (adla04 != 3 & adla04 != 4)
replace enroll_age13to15_other = 0 if adla01 == 3 & age_years >= 13 & age_years <= 15
replace enroll_age13to15_other = 0 if adla04 == 96 & age_years >= 13 & age_years <= 15
replace enroll_age13to15_other = 0 if (adla09 == 1 | adla09a == 1) & age_years >= 13 & age_years <= 15 & (adla04 == 3 | adla04 == 4) 
replace enroll_age13to15_other = 0 if adla09 == 3 & age_years >= 13 & age_years <= 15

//generate any enrollment dummy
gen enroll_7to15 = . 
replace enroll_7to15 = enroll_age7to12 if age_years >= 7 & age_years <= 12
replace enroll_7to15 = enroll_age13to15 if age_years >= 13 & age_years <= 15

//Generate primary school enrollment dummy (non-age restricted)
gen enroll_SD = .
replace enroll_SD = 1 if (adla09 == 1 | adla09a == 1) & (adla04 == 1 | adla04 == 2)
replace enroll_SD = 0 if adla01 == 3
replace enroll_SD = 0 if adla04 == 96
replace enroll_SD = 0 if (adla09 == 1 | adla09a == 1) & adla04 != 1 & adla04 != 2
replace enroll_SD = 0 if adla09 == 3

//Generate primary school enrollment dummy (restricted to ages 7-15 for "pooled" regression panel)
gen enroll_age7to15_SD = enroll_SD
replace enroll_age7to15_SD = . if (age_years < 7 | age_years > 15)

//Generate secondary school enrollment dummy (non-age restricted)
gen enroll_SMP = .
replace enroll_SMP = 1 if (adla09 == 1 | adla09a == 1) & (adla04 == 3 | adla04 == 4)
replace enroll_SMP = 0 if adla01 == 3
replace enroll_SMP = 0 if adla04 == 96
replace enroll_SMP = 0 if (adla09 == 1 | adla09a == 1) & adla04 != 3 & adla04 != 4
replace enroll_SMP = 0 if adla09 == 3

//Generate secondary school enrollment dummy (restricted to ages 7-15 for "pooled" regression panel)
gen enroll_age7to15_SMP = enroll_SMP
replace enroll_age7to15_SMP = . if (age_years < 7 | age_years > 15)

//generate high school enrollment/completion variables 
gen enroll_SMA = .
replace enroll_SMA = 1 if (adla09 == 1 | adla09a == 1) & (adla04 == 5 | adla04 == 6)
replace enroll_SMA = 0 if adla01 == 3
replace enroll_SMA = 0 if adla04 == 96
replace enroll_SMA = 0 if (adla09 == 1 | adla09a == 1) & adla04 != 5 & adla04 != 6
replace enroll_SMA = 0 if (adla09 == 3 | adla09a == 3)

forvalues i = 14/20 {
	gen enroll_SMA_`i' = .
	replace enroll_SMA_`i' = 0 if age_years == `i'
	replace enroll_SMA_`i' = 1 if age_years == `i' & enroll_SMA == 1
}


//dummy for any kind of school enrollment
gen enroll_any = .
replace enroll_any = 1 if (adla09 == 1 | adla09a == 1)
replace enroll_any = 0 if adla01 == 3
replace enroll_any = 0 if adla04 == 96
replace enroll_any = 0 if adla09 == 3

gen enroll_any_15to17 = enroll_any if age_years >= 15 & age_years <= 17

forvalues i = 15/21 {
	gen enroll_any_age`i' = enroll_any if age_years == `i'
}
gen enroll_any_18to21 = enroll_any if age_years >= 18 & age_years <= 21

**************************************

//high school enrollment dummies
gen enroll_SMA_15to17 = .
replace enroll_SMA_15to17 = 0 if age_years >= 15 & age_years <= 17
replace enroll_SMA_15to17 = 1 if age_years >= 15 & age_years <= 17 & enroll_SMA == 1

gen enroll_SMA_18to21 = .
replace enroll_SMA_18to21 = 0 if age_years >= 18 & age_years <= 21
replace enroll_SMA_18to21 = 1 if age_years >= 18 & age_years <= 21 & enroll_SMA == 1

gen completed_SMA = .
replace completed_SMA = 1 if adla09 == 3 & (adla04 == 5 | adla04 == 6) & adla05 == 7
replace completed_SMA = 0 if adla01 == 3
replace completed_SMA = 0 if (adla09 == 1 | adla09a == 1)
replace completed_SMA = 0 if adla04 == 96
replace completed_SMA = 0 if adla09 == 3 & adla04 != 5 & adla04 != 6

//by age 
forvalues i = 17/21 {
	gen completed_SMA_`i' = .
	replace completed_SMA_`i' = 0 if age_years == `i'
	replace completed_SMA_`i' = 1 if age_years == `i' & completed_SMA == 1
}

gen completed_SMA_18to21 = .
replace completed_SMA_18to21 = 0 if age_years >= 18 & age_years <= 21
replace completed_SMA_18to21 = 1 if age_years >= 18 & age_years <= 21 & completed_SMA == 1

//generate high school enrollment conditional on enrollment in school at baseline 
//gen variable for baseline match 
bysort aid (survey_round): gen baseline_match = (survey_round[1] == 0)
bysort aid (survey_round): gen enroll_baseline = (baseline_match == 1 & (adla09[1] == 1 | adla09a[1] == 1))
replace enroll_baseline = . if baseline_match == 0

gen enroll_SMA_cond = .
replace enroll_SMA_cond = 0 if enroll_baseline == 1 
replace enroll_SMA_cond = 1 if enroll_baseline == 1 & enroll_SMA == 1


forvalues i = 14/20 {
	gen enroll_SMA_`i'_cond = .
	replace enroll_SMA_`i'_cond = 0 if age_years == `i' & enroll_baseline == 1
	replace enroll_SMA_`i'_cond = 1 if age_years == `i' & enroll_baseline == 1 & enroll_SMA == 1
}

gen enroll_SMA_15to17_cond = .
replace enroll_SMA_15to17_cond = 0 if age_years >= 15 & age_years <= 17 & enroll_baseline == 1
replace enroll_SMA_15to17_cond = 1 if age_years >= 15 & age_years <= 17 & enroll_baseline == 1 & enroll_SMA == 1

gen completed_SMA_cond = .
replace completed_SMA_cond = 0 if enroll_baseline == 1 
replace completed_SMA_cond = 1 if enroll_baseline == 1 & completed_SMA == 1

forvalues i = 17/21 {
	gen completed_SMA_`i'_cond = .
	replace completed_SMA_`i'_cond = 0 if age_years == `i' & enroll_baseline == 1
	replace completed_SMA_`i'_cond = 1 if age_years == `i' & enroll_baseline == 1 & completed_SMA == 1
}

gen completed_SMA_18to21_cond = .
replace completed_SMA_18to21_cond = 0 if age_years >= 18 & age_years <= 21 & enroll_baseline
replace completed_SMA_18to21_cond = 1 if age_years >= 18 & age_years <= 21 & enroll_baseline == 1 & completed_SMA == 1

//Generate age 7-12 dummy
gen age_7to12 = (age_years >=7 & age_years <=12)
replace age_7to12 = . if age_years == .

//Generate age 13-15 dummy
gen age_13to15 = (age_years >=13 & age_years <=15)
replace age_13to15 = . if age_years == .

//Generate % school attendance in the last week
gen pct_attended_lastweek = days_attend_lastweek/days_open_lastweek
gen age7to12_pct_lastweek = pct_attended_lastweek if age_years>=7 & age_years<=12
gen age13to15_pct_lastweek = pct_attended_lastweek if age_years>=13 & age_years<=15

//Generate % school attendance in last two weeks	
gen pct_attended_twoweeks = adla22_an/adla21_n

//Replace as 100% if more days reported attended than days school open
replace pct_attended_twoweeks = 1 if pct_attended_twoweeks>1 & pct_attended_twoweeks<.

//Generate age-specific percentages
gen age7to12_pct_twoweeks=pct_attended_twoweeks if age_years>=7 & age_years<=12
replace age7to12_pct_twoweeks = 0 if enroll_age7to12 == 0
gen age13to15_pct_twoweeks=pct_attended_twoweeks if age_years>=13 & age_years<=15
replace age13to15_pct_twoweeks = 0 if enroll_age13to15 == 0
gen age7to15_pct_twoweeks=pct_attended_twoweeks if age_years>=7 & age_years<=15
replace age7to15_pct_twoweeks = 0 if enroll_7to15 == 0

//generate percentage conditional on currently attending school (ages 13-15)
gen cond_age7to12_pct_2wks = pct_attended_twoweeks if age_years >= 7 & age_years <= 12 & enroll_age7to12 == 1
gen cond_age13to15_pct_2wks = pct_attended_twoweeks if age_years >= 13 & age_years <= 15 & enroll_age13to15 == 1

//Correcting the attendance percentages to take into kecamatan where days school open is 0 at kec level avg
//(to account for summer vacation)
gen kec_temp = substr(aid,1,3)
bys kec_temp: egen kec_days_open_7to12 = mean(adla21_n) if age_years >= 7 & age_years <= 12
bys kec_temp: egen kec_days_open_13to15 = mean(adla21_n) if age_years >= 13 & age_years <= 15

replace age7to12_pct_twoweeks = . if kec_days_open_7to12 == 0
replace age13to15_pct_twoweeks = . if kec_days_open_13to15 == 0
replace cond_age13to15_pct_2wks = . if kec_days_open_13to15 == 0

drop kec_temp

//Generate dummy for >85% attendance last two weeks (age 7-12)
generate age7to12_85_twoweeks=.
replace age7to12_85_twoweeks=0 if enroll_age7to12==0
replace age7to12_85_twoweeks=1 if pct_attended_twoweeks>=0.85 & pct_attended_twoweeks<. & age_years>=7 & age_years<=12
replace age7to12_85_twoweeks=0 if pct_attended_twoweeks<0.85 & age_years>=7 & age_years<=12

generate age7to12_fullattend_twoweeks = . 
replace age7to12_fullattend_twoweeks = 0 if enroll_age7to12 == 0
replace age7to12_fullattend_twoweeks = 1 if pct_attended_twoweeks >= 1 & pct_attended_twoweeks < . & age_years >= 7 & age_years <= 12
replace age7to12_fullattend_twoweeks = 0 if pct_attended_twoweeks < 1 & age_years >= 7 & age_years <= 12

//Generate dummy for >85% attendance last two weeks (age 13-15)
generate age13to15_85_twoweeks=.
replace age13to15_85_twoweeks=0 if enroll_age13to15==0
replace age13to15_85_twoweeks=1 if pct_attended_twoweeks>=0.85 & pct_attended_twoweeks<. & age_years>=13 & age_years<=15
replace age13to15_85_twoweeks=0 if pct_attended_twoweeks<0.85 & age_years>=13 & age_years<=15

generate age13to15_fullattend_twoweeks = . 
replace age13to15_fullattend_twoweeks = 0 if enroll_age13to15 == 0
replace age13to15_fullattend_twoweeks = 1 if pct_attended_twoweeks >= 1 & pct_attended_twoweeks < . & age_years >= 13 & age_years <= 15
replace age13to15_fullattend_twoweeks = 0 if pct_attended_twoweeks < 1 & age_years >= 13 & age_years <= 15

//Generate dummy for >85% attendance last two weeks (all ages)
generate age7to15_85_twoweeks=.
replace age7to15_85_twoweeks=0 if enroll_7to15 == 0
replace age7to15_85_twoweeks=1 if pct_attended_twoweeks>=0.85 & pct_attended_twoweeks<. & age_years>=7 & age_years<=15
replace age7to15_85_twoweeks=0 if pct_attended_twoweeks<0.85 & age_years>=7 & age_years<=15


//Coding SMP transition rates
//Transitions Wave I
gen SMP_transition = 1 if (adla04 == 3 | adla04 == 4) & adla05 == 1 & adla09 == 1 & adla08g == . & survey_round == 0
replace SMP_transition = 0 if (adla04 == 1 | adla04 == 2) & adla05 == 7 & adla10 == 2007 & survey_round == 0

gen SMP_transition_def2 = 1 if (adla04 == 3 | adla04 == 4) & adla05 == 1 & adla09 == 1 & adla08g == . & survey_round == 0
replace SMP_transition_def2 = 0 if (adla04 == 1 | adla04 == 2) & (adla05 == 6 |adla05 == 7) & adla10 == 2007 & survey_round == 0

//Transitions Wave III
replace SMP_transition = 1 if (adla04 == 3 | adla04 == 4) & adla05 == 1 & (adla09 == 1 | adla09a == 1) & adla08g == . & survey_round == 1
replace SMP_transition = 0 if (adla04 == 1 | adla04 == 2) & adla05 == 7 & adla10 == 2009 & survey_round == 1

replace SMP_transition_def2 = 1 if (adla04 == 3 | adla04 == 4) & adla05 == 1 & adla09 == 1 & adla08g == . & survey_round == 1
replace SMP_transition_def2 = 0 if (adla04 == 1 | adla04 == 2) & (adla05 == 6 |adla05 == 7) & adla10 == 2009 & survey_round == 1

//Transitions Wave IV
replace SMP_transition = 1 if (adla04 == 3 | adla04 == 4) & adla05 == 1 & (adla09 == 1 | adla09a == 1) & adla08g == . & survey_round == 2
replace SMP_transition = 0 if (adla04 == 1 | adla04 == 2) & adla05 == 7 & adla10 == 2013 & survey_round == 2

replace SMP_transition_def2 = 1 if (adla04 == 3 | adla04 == 4) & adla05 == 1 & adla09 == 1 & adla08g == . & survey_round == 2
replace SMP_transition_def2 = 0 if (adla04 == 1 | adla04 == 2) & (adla05 == 6 |adla05 == 7) & adla10 == 2013 & survey_round == 2

//age-specific transition variables
gen SMP_transition_13to15 = SMP_transition if age_years >= 13 & age_years <=15
gen SMP_transition_7to15 = SMP_transition if age_years >= 7 & age_years <= 15

//New definition: EVER transitioned to SMP (exclude 30 obs. where adla09 == 3 but adla09a == 1)
gen SMP_transition_ever = 1 if adla04 >= 3 & adla04 < .
replace SMP_transition_ever = 0 if (adla04 == 1 | adla04 == 2) & adla05 == 7 & adla09 == 3 & adla09a == 3

gen SMP_transition_ever_7to15 = SMP_transition_ever
replace SMP_transition_ever_7to15 = . if (age_years < 7 | age_years > 15)


//Coding school dropout
gen dropout = 1 if adla09 == 3 & (adla10 == 2006 | adla10 == 2007) & survey_round == 0
replace dropout = 1 if adla09 == 3 & (adla10 == 2008 | adla10 == 2009) & survey_round == 1
replace dropout = 1 if adla09 == 3 & (adla10 == 2012 | adla10 == 2013) & survey_round == 2
replace dropout = 0 if adla09 == 1
replace dropout = 0 if adla09 == 3 & (adla04 == 5 | adla04 == 6)
replace dropout = 0 if adla09 == 3 & ((adla04 == 3 | adla04 == 4) & adla05 == 7)
replace dropout = . if adla09 == 3 & ((adla04 == 3 | adla04 == 4) & adla05 == 98)
replace dropout = . if adla09 == 3 & adla04 == 98

//restricted to ages 7-15 for "pooled" regression panel
gen dropout_7to15 = dropout 
replace dropout_7to15 = . if (age_years < 7 | age_years > 15)

*** 6. CODING CHILD LABOR VARIABLES ***

//weeks worked for wage last month
gen weeks_work_wage = atka03_n if atka03_n <= 5 //a handful of excessively large numbers of weeks in Wave I 
gen hrs_wage = atka04_h
gen hrs_wage_frac = atka04_m / 60

//weeks worked for family business
gen weeks_work_fam = atka10_n
gen hrs_fam = atka11_h
gen hrs_fam_frac = atka11_m / 60

//weeks helping out at home
gen weeks_work_home = 13/3 //the "hours working at home" question (atka15) is asked for the last week, not the last month, 
							//so this allows us to extrapolate as best we can for the whole month 
gen hrs_home = atka15_h
gen hrs_home_frac = atka15_m / 60

//Generate total hours worked in past month
//Rule out highly unreasonable entries (>70hrs/week for ages 6-15, >100 hrs/week for ages 16+) 
gen week_wage_hrs = hrs_wage + hrs_wage_frac
replace week_wage_hrs = . if week_wage_hrs > 70 & age_years <= 15
replace week_wage_hrs = . if week_wage_hrs > 100 & age_years > 15 & age_years < .

gen week_fam_hrs = hrs_fam + hrs_fam_frac
replace week_fam_hrs = .  if week_fam_hrs > 70 & age_years <= 15
replace week_fam_hrs = . if week_fam_hrs > 100 & age_years > 15 & age_years < .

gen week_home_hrs = hrs_home + hrs_home_frac
replace week_home_hrs = . if week_home_hrs > 70 & age_years <= 15
replace week_home_hrs = . if week_home_hrs > 100 & age_years > 15 & age_years < .

gen total_wage_hrs = week_wage_hrs * weeks_work_wage 
gen total_fam_hrs =  week_fam_hrs * weeks_work_fam
gen total_home_hrs = week_home_hrs * weeks_work_home

replace total_wage_hrs = 0 if atka01 == 3 | atka02 == 3
replace total_fam_hrs = 0 if atka08 == 3 | atka09 == 3
replace total_home_hrs = 0 if atka14 == 3 | atka15 == 3

//total wage variable
gen total_wage_last_month = atka05_n
gen avg_hourly_wage = total_wage_last_month / total_wage_hrs
gen log_hourly_wage = ln(avg_hourly_wage)
gen log_hourly_wage_7to12 = log_hourly_wage if age_years >= 7 & age_years <= 12
gen log_hourly_wage_13to15 = log_hourly_wage if age_years >= 13 & age_years <= 15
gen log_hourly_wage_16to17 = log_hourly_wage if age_years >= 16 & age_years <= 17
gen log_hourly_wage_18to21 = log_hourly_wage if age_years >= 18 & age_years <= 21

//total hours of labor, including and not including home help 
egen labor_nohome = rowtotal(total_wage_hrs total_fam_hrs), missing 
egen labor_yeshome = rowtotal(total_wage_hrs total_fam_hrs total_home_hrs), missing 
egen labor_wageonly = rowtotal(total_wage_hrs), missing
egen labor_homeonly = rowtotal(total_fam_hrs total_home_hrs), missing 
egen labor_homebusiness = rowtotal(total_fam_hrs), missing
egen labor_helphome = rowtotal(total_home_hrs), missing

//dummy for 20+ and 40+ hours of work in past month, including and not including home help 
gen labor_20_nohome_dummy = . 
replace labor_20_nohome_dummy = 1 if labor_nohome >= 20 & labor_nohome < .
replace labor_20_nohome_dummy = 0 if labor_nohome < 20

gen labor_20_yeshome_dummy = .
replace labor_20_yeshome_dummy = 1 if labor_yeshome >= 20 & labor_yeshome < .
replace labor_20_yeshome_dummy = 0 if labor_yeshome < 20

gen labor_20_wageonly_dummy = .
replace labor_20_wageonly_dummy = 1 if labor_wageonly >= 20 & labor_wageonly < .
replace labor_20_wageonly_dummy = 0 if labor_wageonly < 20

gen labor_20_homeonly_dummy = .
replace labor_20_homeonly_dummy = 1 if labor_homeonly >= 20 & labor_homeonly < .
replace labor_20_homeonly_dummy = 0 if labor_homeonly < 20

gen labor_20_homebusiness_dummy = .
replace labor_20_homebusiness_dummy = 1 if labor_homebusiness >= 20 & labor_homebusiness < .
replace labor_20_homebusiness_dummy = 0 if labor_homebusiness < 20

gen labor_20_helphome_dummy = .
replace labor_20_helphome_dummy = 1 if labor_helphome >= 20 & labor_helphome < .
replace labor_20_helphome_dummy = 0 if labor_helphome < 20

gen labor_40_nohome_dummy = . 
replace labor_40_nohome_dummy = 1 if labor_nohome >= 40 & labor_nohome < .
replace labor_40_nohome_dummy = 0 if labor_nohome < 40

gen labor_40_yeshome_dummy = .
replace labor_40_yeshome_dummy = 1 if labor_yeshome >= 40 & labor_yeshome < .
replace labor_40_yeshome_dummy = 0 if labor_yeshome < 40

gen labor_40_wageonly_dummy = .
replace labor_40_wageonly_dummy = 1 if labor_wageonly >= 40 & labor_wageonly < .
replace labor_40_wageonly_dummy = 0 if labor_wageonly < 40

gen labor_40_homeonly_dummy = .
replace labor_40_homeonly_dummy = 1 if labor_homeonly >= 40 & labor_homeonly < .
replace labor_40_homeonly_dummy = 0 if labor_homeonly < 40

gen labor_40_homebusiness_dummy = .
replace labor_40_homebusiness_dummy = 1 if labor_homebusiness >= 40 & labor_homebusiness < .
replace labor_40_homebusiness_dummy = 0 if labor_homebusiness < 40

gen labor_40_helphome_dummy = .
replace labor_40_helphome_dummy = 1 if labor_helphome >= 40 & labor_helphome < .
replace labor_40_helphome_dummy = 0 if labor_helphome < 40

//dummy for any labor in past month
gen any_labor_wageonly_dummy = .
replace any_labor_wageonly_dummy = 1 if labor_wageonly > 0
replace any_labor_wageonly_dummy = 1 if (atka02 == 1)
replace any_labor_wageonly_dummy = 0 if labor_wageonly == 0
replace any_labor_wageonly_dummy = 0 if (atka02 == 3)

gen any_labor_nohome_dummy = .
replace any_labor_nohome_dummy = 1 if labor_nohome > 0
replace any_labor_nohome_dummy = 1 if (atka02 == 1 | atka09 == 1)
replace any_labor_nohome_dummy = 0 if labor_nohome == 0
replace any_labor_nohome_dummy = 0 if (atka02 == 3 & atka09 == 3)

gen any_labor_yeshome_dummy = .
replace any_labor_yeshome_dummy = 1 if labor_yeshome > 0
replace any_labor_yeshome_dummy = 1 if (atka02 == 1 | atka09 == 1 | atka14 == 1)
replace any_labor_yeshome_dummy = 0 if labor_yeshome == 0
replace any_labor_yeshome_dummy = 0 if (atka02 == 3 & atka09 == 3 & atka14 == 3)

gen any_labor_homeonly_dummy = .
replace any_labor_homeonly_dummy = 1 if labor_homeonly > 0
replace any_labor_homeonly_dummy = 1 if (atka09 == 1 | atka14 == 1)
replace any_labor_homeonly_dummy = 0 if labor_homeonly == 0
replace any_labor_homeonly_dummy = 0 if (atka09 == 3 & atka14 == 3)

gen any_labor_homebusiness_dummy = .
replace any_labor_homebusiness_dummy = 1 if labor_homeonly > 0
replace any_labor_homebusiness_dummy = 1 if (atka09 == 1)
replace any_labor_homebusiness_dummy = 0 if labor_homeonly == 0
replace any_labor_homebusiness_dummy = 0 if (atka09 == 3)

gen any_labor_helphome_dummy = .
replace any_labor_helphome_dummy = 1 if labor_homeonly > 0
replace any_labor_helphome_dummy = 1 if (atka14 == 1)
replace any_labor_helphome_dummy = 0 if labor_homeonly == 0
replace any_labor_helphome_dummy = 0 if (atka14 == 3)

//age-specific work dummies 
gen any_work_7to12 = any_labor_yeshome_dummy if age_years >= 7 & age_years <= 12
gen any_work_13to15 = any_labor_yeshome_dummy if age_years >= 13 & age_years <= 15

gen wage_work_7to12 = any_labor_nohome_dummy if age_years >= 7 & age_years <= 12
gen wage_work_13to15 = any_labor_nohome_dummy if age_years >= 13 & age_years <= 15

gen any_work_20hrs_7to12 = labor_20_yeshome_dummy if age_years >= 7 & age_years <= 12
gen any_work_20hrs_13to15 = labor_20_yeshome_dummy if age_years >= 13 & age_years <= 15

gen wage_work_20hrs_7to12 = labor_20_nohome_dummy if age_years >= 7 & age_years <= 12
gen wage_work_20hrs_13to15 = labor_20_nohome_dummy if age_years >= 13 & age_years <= 15

gen any_work_40hrs_7to12 = labor_40_yeshome_dummy if age_years >= 7 & age_years <= 12
gen any_work_40hrs_13to15 = labor_40_yeshome_dummy if age_years >= 13 & age_years <= 15

gen wage_work_40hrs_7to12 = labor_40_nohome_dummy if age_years >= 7 & age_years <= 12
gen wage_work_40hrs_13to15 = labor_40_nohome_dummy if age_years >= 13 & age_years <= 15


//7 to 12
gen wageonly_7to12 = any_labor_wageonly_dummy if age_years >= 7 & age_years <= 12
gen wageonly_20hrs_7to12 = labor_20_wageonly_dummy if age_years >= 7 & age_years <= 12
gen wageonly_40hrs_7to12 = labor_40_wageonly_dummy if age_years >= 7 & age_years <= 12

gen homeonly_7to12 = any_labor_homeonly_dummy if age_years >= 7 & age_years <= 12
gen homeonly_20hrs_7to12 = labor_20_homeonly_dummy if age_years >= 7 & age_years <= 12
gen homeonly_40hrs_7to12 = labor_40_homeonly_dummy if age_years >= 7 & age_years <= 12

gen homebusiness_7to12 = any_labor_homebusiness_dummy if age_years >= 7 & age_years <= 12
gen homebusiness_20hrs_7to12 = labor_20_homebusiness_dummy if age_years >= 7 & age_years <= 12
gen homebusiness_40hrs_7to12 = labor_40_homebusiness_dummy if age_years >= 7 & age_years <= 12

gen helphome_7to12 = any_labor_helphome_dummy if age_years >= 7 & age_years <= 12
gen helphome_20hrs_7to12 = labor_20_helphome_dummy if age_years >= 7 & age_years <= 12
gen helphome_40hrs_7to12 = labor_40_helphome_dummy if age_years >= 7 & age_years <= 12

//13 to 15
gen wageonly_13to15 = any_labor_wageonly_dummy if age_years >= 13 & age_years <= 15
gen wageonly_20hrs_13to15 = labor_20_wageonly_dummy if age_years >= 13 & age_years <= 15
gen wageonly_40hrs_13to15 = labor_40_wageonly_dummy if age_years >= 13 & age_years <= 15

gen homeonly_13to15 = any_labor_homeonly_dummy if age_years >= 13 & age_years <= 15
gen homeonly_20hrs_13to15 = labor_20_homeonly_dummy if age_years >= 13 & age_years <= 15
gen homeonly_40hrs_13to15 = labor_40_homeonly_dummy if age_years >= 13 & age_years <= 15

gen homebusiness_13to15 = any_labor_homebusiness_dummy if age_years >= 13 & age_years <= 15
gen homebusiness_20hrs_13to15 = labor_20_homebusiness_dummy if age_years >= 13 & age_years <= 15
gen homebusiness_40hrs_13to15 = labor_40_homebusiness_dummy if age_years >= 13 & age_years <= 15

gen helphome_13to15 = any_labor_helphome_dummy if age_years >= 13 & age_years <= 15
gen helphome_20hrs_13to15 = labor_20_helphome_dummy if age_years >= 13 & age_years <= 15
gen helphome_40hrs_13to15 = labor_40_helphome_dummy if age_years >= 13 & age_years <= 15

//16 to 17
gen wage_work_16to21 = any_labor_nohome_dummy if age_years >= 16 & age_years <= 21
gen wage_work_20hrs_16to21 = labor_20_nohome_dummy if age_years >= 16 & age_years <= 21
gen wage_work_40hrs_16to21 = labor_40_nohome_dummy if age_years >= 16 & age_years <= 21

gen wage_work_16to17 = any_labor_nohome_dummy if age_years >= 16 & age_years <= 17
gen wage_work_20hrs_16to17 = labor_20_nohome_dummy if age_years >= 16 & age_years <= 17
gen wage_work_40hrs_16to17 = labor_40_nohome_dummy if age_years >= 16 & age_years <= 17

gen wageonly_16to17 = any_labor_wageonly_dummy if age_years >= 16 & age_years <= 17
gen wageonly_20hrs_16to17 = labor_20_wageonly_dummy if age_years >= 16 & age_years <= 17
gen wageonly_40hrs_16to17 = labor_40_wageonly_dummy if age_years >= 16 & age_years <= 17

gen homeonly_16to17 = any_labor_homeonly_dummy if age_years >= 16 & age_years <= 17
gen homeonly_20hrs_16to17 = labor_20_homeonly_dummy if age_years >= 16 & age_years <= 17
gen homeonly_40hrs_16to17 = labor_40_homeonly_dummy if age_years >= 16 & age_years <= 17

gen homebusiness_16to17 = any_labor_homebusiness_dummy if age_years >= 16 & age_years <= 17
gen homebusiness_20hrs_16to17 = labor_20_homebusiness_dummy if age_years >= 16 & age_years <= 17
gen homebusiness_40hrs_16to17 = labor_40_homebusiness_dummy if age_years >= 16 & age_years <= 17

gen helphome_16to17 = any_labor_helphome_dummy if age_years >= 16 & age_years <= 17
gen helphome_20hrs_16to17 = labor_20_helphome_dummy if age_years >= 16 & age_years <= 17
gen helphome_40hrs_16to17 = labor_40_helphome_dummy if age_years >= 16 & age_years <= 17

//18 to 21
gen wage_work_18to21 = any_labor_nohome_dummy if age_years >= 18 & age_years <= 21
gen wage_work_20hrs_18to21 = labor_20_nohome_dummy if age_years >= 18 & age_years <= 21
gen wage_work_40hrs_18to21 = labor_40_nohome_dummy if age_years >= 18 & age_years <= 21

gen wageonly_18to21 = any_labor_wageonly_dummy if age_years >= 18 & age_years <= 21
gen wageonly_20hrs_18to21 = labor_20_wageonly_dummy if age_years >= 18 & age_years <= 21
gen wageonly_40hrs_18to21 = labor_40_wageonly_dummy if age_years >= 18 & age_years <= 21

gen homeonly_18to21 = any_labor_homeonly_dummy if age_years >= 18 & age_years <= 21
gen homeonly_20hrs_18to21 = labor_20_homeonly_dummy if age_years >= 18 & age_years <= 21
gen homeonly_40hrs_18to21 = labor_40_homeonly_dummy if age_years >= 18 & age_years <= 21

gen homebusiness_18to21 = any_labor_homebusiness_dummy if age_years >= 18 & age_years <= 21
gen homebusiness_20hrs_18to21 = labor_20_homebusiness_dummy if age_years >= 18 & age_years <= 21
gen homebusiness_40hrs_18to21 = labor_40_homebusiness_dummy if age_years >= 18 & age_years <= 21

gen helphome_18to21 = any_labor_helphome_dummy if age_years >= 18 & age_years <= 21
gen helphome_20hrs_18to21 = labor_20_helphome_dummy if age_years >= 18 & age_years <= 21
gen helphome_40hrs_18to21 = labor_40_helphome_dummy if age_years >= 18 & age_years <= 21

//by-year breakdown for older
forvalues i = 18/21 {
	gen wage_work_age`i' = any_labor_nohome_dummy if age_years == `i'
	gen wage_work_20hrs_age`i' = labor_20_nohome_dummy if age_years == `i' 
	gen wage_work_40hrs_age`i' = labor_40_nohome_dummy if age_years == `i' 

	gen wageonly_age`i' = any_labor_wageonly_dummy if age_years == `i'
	gen wageonly_20hrs_age`i' = labor_20_wageonly_dummy if age_years == `i' 
	gen wageonly_40hrs_age`i' = labor_40_wageonly_dummy if age_years == `i' 
}


//age-specific total hours of labor 
gen total_hours_any_7to12 = labor_yeshome if age_years >= 7 & age_years <= 12
gen total_hours_any_13to15 = labor_yeshome if age_years >= 13 & age_years <= 15

gen total_hours_wage_7to12 = labor_nohome if age_years >= 7 & age_years <= 12
gen total_hours_wage_13to15 = labor_nohome if age_years >= 13 & age_years <= 15


//interaction variable for enrolled in school * not working for wage 
gen not_working_wage = (any_labor_wageonly_dummy == 0) if !missing(any_labor_wageonly_dummy)
gen enroll_nowagework_7to12 = enroll_age7to12 * not_working_wage
gen enroll_nowagework_13to15 = enroll_age13to15 * not_working_wage
gen enroll_andwagework_7to12 = enroll_age7to12 * any_labor_wageonly_dummy
gen enroll_andwagework_13to15 = enroll_age13to15 * any_labor_wageonly_dummy



*** 7. MERGING HOUSEHOLD-LEVEL COVARIATES ***
//create rid variable to match children with households from HH file
gen rid = substr(aid, 1, 3) + "1" + substr(aid, 5, 5)

tempfile child_allwaves_coded
save `child_allwaves_coded', replace 

//load master household dataset 
cd "`PKH'"
use "data/coded/household_allwaves_master.dta", clear 

//keep only relevant household covariates 
keep rid* survey_round received* pkh* L07 hh_head_education hh_head_agr ///
	 hh_head_serv hh_educ* roof_type* wall_type* floor_type* clean_water own_latrine ///
	 square_latrine own_septic_tank electricity_PLN logpcexp* hhsize_ln *baseline_nm *miss ///
	 kabupaten kecamatan province split_indicator lack* low_* hh_surveyed_endline num_*_pc_bl total_*_pc_bl non_pkh_hh_cct_kec

tempfile hh_covariates
save `hh_covariates', replace

//merge 
use `child_allwaves_coded', clear

//deal with anomalous observations with "13" for in RT code 
gen temp_code = substr(aid_w4,12,2)
gen temp_indicator = 1 if temp_code == "13"
//generate rid_merge variable to ensure merge to correct household (accounts for HH splits after baseline)
gen rid_merge = substr(aid, 1, 3) + "1" + substr(aid,5,5) if survey_round == 0
replace rid_merge = substr(aid_w3, 1, 3) + "1" + substr(aid_w3,5,9) if survey_round == 1
replace rid_merge = substr(aid_w4, 1, 3) + "1" + substr(aid_w4,5,13) if survey_round == 2 & temp_indicator != 1
replace rid_merge = substr(aid_w4, 1, 3) + "1" + substr(aid_w4,5,7) + "00" + substr(aid_w4,14,4) if survey_round == 2 & temp_indicator == 1

merge m:1 rid_merge survey_round using `hh_covariates'
tab _merge
drop if _merge == 2
drop _merge 


*** MERGE WITH ATTRITION FILE TO IDENTIFY WHICH CHILDREN IN WAVE III ARE TRACKED DOWN IN WAVE IV
//create dataset with only children surveyed at endline 
preserve
keep if survey_round == 2
gen aid_attrition_merge = substr(aid_w4, 1, 7) + substr(aid_w4, 10, 6) + substr(aid_w4, 18, 2) if temp_indicator != 1
replace aid_attrition_merge = substr(aid_w4, 1, 7) + substr(aid_w4, 10, 2) + "00" + substr(aid_w4, 14, 2) + substr(aid_w4, 18, 2) if temp_indicator == 1
drop if acov3 == 3
//change survey round variable for merge 
replace survey_round = 1
keep aid_attrition_merge survey_round age_years 
duplicates drop aid_attrition_merge, force //1 erroneous duplicate
//age at endline 
gen age_years_w4 = age_years 

tempfile attrition_merge 
save `attrition_merge', replace 
restore

//merge 
gen aid_attrition_merge = aid_w3
merge m:1 aid_attrition_merge survey_round using `attrition_merge'
tab _merge 
drop if _merge == 2

//generate indicator for kids in Wave III who show up in Wave IV 
gen surveyed_endline = (_merge == 3) if survey_round == 1
drop _merge 



sort aid survey_round 
//save
compress
cd "`PKH'data/coded"
save "child6to15_allwaves_master", replace


