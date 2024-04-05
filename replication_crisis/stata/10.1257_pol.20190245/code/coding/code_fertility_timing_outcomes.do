


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
local latex = "`output'/latex"
local TARGET2 = "$TARGET2"


//Set working directory:
cd "`pkhdata'"


*** 3. ASSEMBLE DATASETS FROM ACROSS WAVES ***
*Wave III
use "Wave I/Data/I_fin.dta", clear 
duplicates drop 

//generate survey date variable to compare child birth dates 
gen survey_date_w1 = mdy(iivw_mm_1, iivw_dd_1, iivw_yy_1) if ihkw_1 == 1 | ihkw_1 == 2
gen survey_month_w1 = ym(iivw_yy_1, iivw_mm_1) if ihkw_1 == 1 | ihkw_1 == 2
gen survey_year_w1 = iivw_yy_1 if ihkw_1 == 1 | ihkw_1 == 2

replace survey_date_w1 = mdy(iivw_mm_2, iivw_dd_2, iivw_yy_2) if survey_date_w1 == . & (ihkw_2==1 | ihkw_2==2)
replace survey_month_w1 = ym(iivw_yy_2, iivw_mm_2) if survey_month_w1 == . & (ihkw_2==1 | ihkw_2==2)
replace survey_year_w1 = iivw_yy_2 if survey_year_w1 == . & (ihkw_2 == 1 | ihkw_2 == 2)

replace survey_date_w1 = mdy(iivw_mm_3, iivw_dd_3, iivw_yy_3) if survey_date== . & (ihkw_3==1 | ihkw_3==2)
replace survey_month_w1 = ym(iivw_yy_3, iivw_mm_3) if survey_month_w1 == . & (ihkw_3==1 | ihkw_3==2)
replace survey_year_w1 = iivw_yy_3 if survey_year_w1 == . & (ihkw_3 == 1 | ihkw_3 == 2)


//code relevant outcomes 
gen num_kids_in_hh_w1 = irh04_c
replace num_kids_in_hh_w1 = 0 if missing(irh04_c)
gen num_kids_alive_away_w1 = irh06_c 
replace num_kids_alive_away_w1 = 0 if missing(irh06_c)
gen num_kids_born_died_w1 = irh12_c
replace num_kids_born_died_w1 = 0 if missing(irh12_c)
gen num_stillbirths_w1 = irh18
replace num_stillbirths_w1 = 0 if missing(irh18)
gen num_miscarriages_w1 = irh22
replace num_miscarriages_w1 = 0 if missing(irh22)
gen currently_pregnant_w1 = (irh26 == 1)

gen total_births_all_w1 = num_kids_in_hh_w1 + num_kids_alive_away_w1 + num_kids_born_died_w1 + num_stillbirths_w1 + num_miscarriages_w1 + currently_pregnant_w1
gen total_births_nosbmc_w1 = num_kids_in_hh_w1 + num_kids_alive_away_w1 + num_kids_born_died_w1 + currently_pregnant_w1

//survey round indicator 
gen survey_round = 0
//iid variable to match between waves 
gen iid_w1_w3 = iid 

tempfile mother_births_w1
save `mother_births_w1', replace 


*Wave III
//start with household roster 
use "Wave III/Data/R_AR_01_fin.dta", clear 
duplicates drop 

//create iid variable to merge with CURRENT wave's I.dta, as well as previous wave's 
tostring rar00 rar00_07, format(%02.0f) replace
gen iid_w3 = substr(rid, 1, 3) + "2" + substr(rid, 5, 9) + rar00 

 
gen iid_w1_w3 = substr(rid, 1, 3) + "2" + substr(rid, 5, 5) + rar00 if rar00_07 == "00"
replace iid_w1_w3 = substr(rid, 1, 3) + "2" + substr(rid, 5, 5) + rar00_07 if rar00_07 != "00"
gen split_indicator = (substr(rid, 12, 2) != "00")
replace iid_w1_w3 = "" if rar00_07 == "00" & split_indicator == 1 & missing(rar01a)
replace iid_w1_w3 = "" if substr(rid, 10, 2) != "00" //households that split in Wave II: not relevant for our analysis 

tempfile roster_w3 
save `roster_w3', replace 

//Main mother dataset 
use "Wave III/Data/i_fin.dta", clear 
duplicates drop 

//merge with roster data 
gen iid_w3 = iid 
merge 1:1 iid_w3 using `roster_w3'
tab _merge  
drop if _merge == 2
drop _merge 

//code relevant outcomes 
gen num_kids_in_hh_w3 = irh04_c
replace num_kids_in_hh_w3 = 0 if missing(irh04_c)
gen num_kids_alive_away_w3 = irh06_c 
replace num_kids_alive_away_w3 = 0 if missing(irh06_c)
gen num_kids_born_died_w3 = irh12_c
replace num_kids_born_died_w3 = 0 if missing(irh12_c)
gen num_stillbirths_w3 = irh18
replace num_stillbirths_w3 = 0 if missing(irh18)
gen num_miscarriages_w3 = irh22
replace num_miscarriages_w3 = 0 if missing(irh22)
gen currently_pregnant_w3 = (irh26 == 1)

gen total_births_all_w3 = num_kids_in_hh_w3 + num_kids_alive_away_w3 + num_kids_born_died_w3 + num_stillbirths_w3 + num_miscarriages_w3 + currently_pregnant_w3
gen total_births_nosbmc_w3 = num_kids_in_hh_w3 + num_kids_alive_away_w3 + num_kids_born_died_w3 + currently_pregnant_w3

//survey round indicator 
gen survey_round = 1
//iid variable to match between waves 
gen iid_w3_w4 = iid_w3 


tempfile mother_births_w3
save `mother_births_w3', replace 





*Wave IV
//start with household roster 
use "Wave IV/R_AR_01.dta", clear 
duplicates drop 

//create iid variable to merge with CURRENT wave's I.dta
tostring rar00 rar00_09, format(%02.0f) replace
gen iid_roster_merge = substr(rid, 1, 3) + "2" + substr(rid, 5, 13) + rar00

keep rid rar00 rar00_09 iid_roster_merge


tempfile roster_w4
save `roster_w4', replace 

//Main mother dataset 
use "Wave IV/I.dta", clear 
duplicates drop 

//merge with roster data 
tostring icov1_cd, format(%02.0f) replace 
replace icov1_cd = "04" if iid == "6082001000400000004" //one entry error
gen iid_roster_merge = substr(iid, 1, 17) + icov1_cd 

//merge with roster data 
gen iid_w4 = iid 
merge 1:1 iid_roster_merge using `roster_w4'
tab _merge  
drop if _merge == 2
drop _merge 

//code relevant outcomes 
gen num_kids_in_hh_w4 = irh04_c
replace num_kids_in_hh_w4 = 0 if missing(irh04_c)
gen num_kids_alive_away_w4 = irh06_c 
replace num_kids_alive_away_w4 = 0 if missing(irh06_c)
gen num_kids_born_died_w4 = irh12_c
replace num_kids_born_died_w4 = 0 if missing(irh12_c)
gen num_stillbirths_w4 = irh18
replace num_stillbirths_w4 = 0 if missing(irh18)
gen num_miscarriages_w4 = irh22
replace num_miscarriages_w4 = 0 if missing(irh22)
gen currently_pregnant_w4 = (irh26 == 1)

gen total_births_all_w4 = num_kids_in_hh_w4 + num_kids_alive_away_w4 + num_kids_born_died_w4 + num_stillbirths_w4 + num_miscarriages_w4 + currently_pregnant_w4
gen total_births_nosbmc_w4 = num_kids_in_hh_w4 + num_kids_alive_away_w4 + num_kids_born_died_w4 + currently_pregnant_w4

//survey round indicator 
gen survey_round = 2
//iid variable to match between waves 
gen iid_w3_w4 = substr(iid_w4, 1, 7) + substr(iid_w4, 10, 6) + substr(iid_w4, 18, 2) 
replace iid_w3_w4 = "" if rar00_09 == "00"
replace iid_w3_w4 = "" if icov3 == 3 



*** APPEND ALL WAVES AND ALIGN BIRTH OUTCOMES ***
append using `mother_births_w1' `mother_births_w3', force 
sort iid survey_round

//Wave I - III
bysort iid_w1_w3 (survey_round): gen match_w1_w3 = (_N == 2) if survey_round != 2
bysort iid_w1_w3 (survey_round): carryforward total_births_all_w1 total_births_nosbmc_w1 num_kids_in_hh_w1 num_kids_alive_away_w1 num_kids_born_died_w1 num_stillbirths_w1 ///
								 survey_date_w1 survey_month_w1 survey_year_w1 num_miscarriages_w1 currently_pregnant_w1 if match_w1_w3 == 1, replace

//Wave III - IV 
bysort iid_w3_w4 (survey_round): gen match_w3_w4 = (_N == 2) if survey_round != 0
bysort iid_w3_w4 (survey_round): carryforward total_births_all_w3 total_births_nosbmc_w3 num_kids_in_hh_w3 num_kids_alive_away_w3 num_kids_born_died_w3 num_stillbirths_w3 num_miscarriages_w3 currently_pregnant_w3 if match_w3_w4 == 1, replace
bysort iid_w3_w4 (survey_round): carryforward match_w1_w3 if match_w3_w4 == 1, replace
bysort iid_w3_w4 (survey_round): carryforward total_births_all_w1 total_births_nosbmc_w1 num_kids_in_hh_w1 num_kids_alive_away_w1 num_kids_born_died_w1 num_stillbirths_w1 ///
								 survey_date_w1 survey_month_w1 survey_year_w1 num_miscarriages_w1 currently_pregnant_w1 if match_w1_w3 == 1 & match_w3_w4 == 1, replace


//birth outcomes 
gen births_since_baseline_all = total_births_all_w3 - total_births_all_w1 if survey_round == 1 & match_w1_w3 == 1 
replace births_since_baseline_all = total_births_all_w4 - total_births_all_w1 if survey_round == 2 & match_w3_w4 == 1 & match_w1_w3 == 1

gen births_since_baseline_nosbmc = total_births_nosbmc_w3 - total_births_nosbmc_w1 if survey_round == 1 & match_w1_w3 == 1 
replace births_since_baseline_nosbmc = total_births_nosbmc_w4 - total_births_nosbmc_w1 if survey_round == 2 & match_w3_w4 == 1 & match_w1_w3 == 1

tempfile mother_allwaves
save `mother_allwaves', replace 

*** CODING PKH RANDOMIZATION/TREATMENT STATUS ***
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
use `mother_allwaves', clear

//generate ea string
gen ea = substr(iid, 1, 3)

//merge with kecamatan treatment list 
merge m:1 ea using `ea_treatment_list', generate(merge_treat)
tab merge_treat
//only keep observations from original 360 kecamatans (L07 is non-missing)
keep if merge_treat == 3
drop merge_treat
sort iid survey_round


*** 6. MERGING HOUSEHOLD-LEVEL COVARIATES ***
tempfile mother_allwaves_coded
save `mother_allwaves_coded', replace 

//load master household dataset 
cd "`PKH'"
use "data/coded/household_allwaves_master.dta", clear 

//keep only relevant household covariates 
keep rid* survey_round received* pkh* L07 hh_head_education hh_head_agr ///
	 hh_head_serv hh_educ* roof_type* wall_type* floor_type* clean_water own_latrine ///
	 square_latrine own_septic_tank electricity_PLN logpcexp* hhsize_ln *baseline_nm *miss ///
	 kabupaten kecamatan province lack* low*

tempfile hh_covariates
save `hh_covariates', replace

//merge 
use `mother_allwaves_coded', clear 
//generate rid_merge variable to merge to correct household (accounts for HH splits after baseline)
gen rid_merge = substr(iid, 1, 3) + "1" + substr(iid,5,5) if survey_round == 0
replace rid_merge = substr(iid_w3, 1, 3) + "1" + substr(iid_w3,5,9) if survey_round == 1
replace rid_merge = substr(iid_w4, 1, 3) + "1" + substr(iid_w4,5,13) if survey_round == 2 


merge m:1 rid_merge survey_round using `hh_covariates', generate(_merge_hh)
tab _merge_hh
drop if _merge_hh == 2
drop _merge_hh


tempfile fertility_allwaves_coded 
save `fertility_allwaves_coded', replace 






***EXAMINE FULL ROSTER OF MOTHER'S CHILDREN: WAVE III ***
cd "`pkhdata'"
*Mother's birth history of kids living outside household 
use "Wave III/Data/I_RH07_fin.dta", clear 
duplicates drop 

//indicator for alive, living outside house 
gen alive_live_away = 1

drop irh07x irh09 irh10

tempfile alive_live_away_w3
save `alive_live_away_w3', replace 

*History of kids born but then died 
use "Wave III/Data/I_RH13_fin.dta", clear 
duplicates drop

//(no unfilled lines)
keep iid irh13_row irh13_mm irh13_yy

//indicator for born alive but then died 
gen born_alive_died = 1

tempfile born_alive_died_w3
save `born_alive_died_w3', replace 

*History of stillbirths
use "Wave III/Data/I_RH19_fin.dta", clear 
duplicates drop 

//drop unfilled lines 
keep iid irh19_row irh19_mm irh19_yy

//indicator for stillbirth 
gen stillbirth = 1

tempfile stillbirths_w3
save `stillbirths_w3', replace 

*History of miscarriages 
use "Wave III/Data/I_RH23_fin.dta"
duplicates drop 

drop irh23a irh24 

//indicator for miscarriage 
gen miscarriage = 1

tempfile miscarriages_w3
save `miscarriages_w3', replace 



***Open main household roster 
use "Wave III/Data/R_AR_01_fin.dta", clear 
duplicates drop 

//for individuals whose parents live in the household, create variable indicating iid of mother (for merge with 1B data)
tostring rar06, gen(mother_serial) format(%02.0f)
gen iid = substr(rid, 1, 3) + "2" + substr(rid, 5, 9) + mother_serial if !missing(mother_serial)

//indicator for actual roster obs 
gen roster_member = 1

//append using birth histories from above 
append using `alive_live_away_w3' `born_alive_died_w3' `stillbirths_w3' `miscarriages_w3'

//create single variable on which birth dates can be sorted 
gen birth_year_sort = rar04_yy
gen birth_month_sort = rar04_mm
gen birth_day_sort = rar04_dd 

replace birth_year_sort = irh08_yy if alive_live_away == 1
replace birth_month_sort = irh08_mm if alive_live_away == 1
replace birth_day_sort = irh08_dd if alive_live_away == 1

replace birth_year_sort = irh13_yy if born_alive_died == 1
replace birth_month_sort = irh13_mm	if born_alive_died == 1

replace birth_year_sort = irh19_yy if stillbirth == 1
replace birth_month_sort = irh19_mm if stillbirth == 1

replace birth_year_sort = irh23_yy if miscarriage == 1
replace birth_month_sort = irh23_mm if miscarriage == 1

//survey round indicator 
gen survey_round = 1

tempfile birth_roster_w3
save `birth_roster_w3', replace 






*** 4. WAVE IV: ASSEMBLE ALL BIRTHS IN HOUSEHOLD ***
*Mother's birth history of kids living outside household 
use "Wave IV/I_RH07.dta", clear 
duplicates drop 
drop if missing(irh07_row)
destring irh08_dd irh08_mm irh08_yy, replace 
//indicator for alive, living outside house 
gen alive_live_away = 1

drop irh07x irh09 irh10

tempfile alive_live_away_w4
save `alive_live_away_w4', replace 

*History of kids born but then died 
use "Wave IV/I_RH13.dta", clear 
duplicates drop

//drop unfilled lines 
drop if missing(irh13_row)
destring irh13_mm irh13_yy, replace 
keep iid irh13_row irh13_mm irh13_yy

//indicator for born alive but then died 
gen born_alive_died = 1

tempfile born_alive_died_w4
save `born_alive_died_w4', replace 

*History of stillbirths
use "Wave IV/I_RH19.dta", clear 
duplicates drop 

//drop unfilled lines 
drop if missing(irh19_row)
destring irh19_mm irh19_yy, replace 
keep iid irh19_row irh19_mm irh19_yy

//indicator for stillbirth 
gen stillbirth = 1

tempfile stillbirths_w4
save `stillbirths_w4', replace 

*History of miscarriages 
use "Wave IV/I_RH23.dta"
duplicates drop 

//drop unfilled lines 
drop if missing(irh23_row)
destring irh23_mm irh23_yy, replace 
drop irh23a irh24 

//indicator for miscarriage 
gen miscarriage = 1

tempfile miscarriages_w4
save `miscarriages_w4', replace 



***Open main household roster 
use "Wave IV/R_AR_01.dta", clear 
duplicates drop 

//for individuals whose parents live in the household, create variable indicating iid of mother (for merge with 1B data)
tostring rar06, gen(mother_serial) format(%02.0f)
gen iid = substr(rid, 1, 3) + "2" + substr(rid, 5, 13) + mother_serial if !missing(mother_serial)

//indicator for actual roster obs 
gen roster_member = 1

//append using birth histories from above 
append using `alive_live_away_w4' `born_alive_died_w4' `stillbirths_w4' `miscarriages_w4'

//create single variable on which birth dates can be sorted 
gen birth_year_sort = rar04_yy
gen birth_month_sort = rar04_mm
gen birth_day_sort = rar04_dd 

replace birth_year_sort = irh08_yy if alive_live_away == 1
replace birth_month_sort = irh08_mm if alive_live_away == 1
replace birth_day_sort = irh08_dd if alive_live_away == 1

replace birth_year_sort = irh13_yy if born_alive_died == 1
replace birth_month_sort = irh13_mm	if born_alive_died == 1

replace birth_year_sort = irh19_yy if stillbirth == 1
replace birth_month_sort = irh19_mm if stillbirth == 1

replace birth_year_sort = irh23_yy if miscarriage == 1
replace birth_month_sort = irh23_mm if miscarriage == 1


//survey round indicator 
gen survey_round = 2

tempfile birth_roster_w4
save `birth_roster_w4', replace 




//open listing of mothers and merge with birth rosters 
use `fertility_allwaves_coded', clear 

merge 1:m iid survey_round using `birth_roster_w3', keepusing(birth_day_sort birth_month_sort birth_year_sort roster_member alive_live_away born_alive_died stillbirth miscarriage)
tab _merge  
drop if _merge == 2
drop _merge 

keep if survey_round == 1

//code kids' birth dates in relation to survey date 
replace birth_day_sort = 15 if missing(birth_day_sort) & !missing(birth_year_sort) & !missing(birth_month_sort)
gen birth_date_numeric = mdy(birth_month_sort, birth_day_sort, birth_year_sort)
gen birth_month_numeric = ym(birth_year_sort, birth_month_sort)

gen birth_since_baseline_all_v2 = (birth_date_numeric > survey_date_w1) if !missing(survey_date_w1) & !missing(birth_date_numeric)
replace birth_since_baseline_all_v2 = 1 if birth_month_numeric > survey_month_w1 & missing(survey_date_w1) & !missing(survey_month_w1) & !missing(birth_month_numeric)
replace birth_since_baseline_all_v2 = 1 if birth_year_sort > survey_year_w1 & missing(survey_date_w1) & missing(survey_month_w1) & !missing(survey_year_w1) & !missing(birth_year_sort)

gen birth_since_baseline_nosbmc_v2 = (birth_date_numeric > survey_date_w1 & stillbirth != 1 & miscarriage != 1) if !missing(survey_date_w1) & !missing(birth_date_numeric)
replace birth_since_baseline_nosbmc_v2 = 1 if birth_month_numeric > survey_month_w1 & stillbirth != 1 & miscarriage != 1 & missing(survey_date_w1) & !missing(survey_month_w1) & !missing(birth_month_numeric)
replace birth_since_baseline_nosbmc_v2 = 1 if birth_year_sort > survey_year_w1 & stillbirth != 1 & miscarriage != 1 & missing(survey_date_w1) & missing(survey_month_w1) & !missing(survey_year_w1) & !missing(birth_year_sort)

collapse (sum) births_since_baseline_all_v2=birth_since_baseline_all_v2 births_since_baseline_nosbmc_v2=birth_since_baseline_nosbmc_v2, by(iid survey_round)
tempfile births_v2_w3
save `births_v2_w3', replace 

*Wave IV
use `fertility_allwaves_coded', clear 

merge 1:m iid survey_round using `birth_roster_w4', keepusing(birth_day_sort birth_month_sort birth_year_sort roster_member alive_live_away born_alive_died stillbirth miscarriage)
tab _merge  
drop if _merge == 2
drop _merge 

keep if survey_round == 2

//code kids' birth dates in relation to survey date 
replace birth_day_sort = 15 if missing(birth_day_sort) & !missing(birth_year_sort) & !missing(birth_month_sort)
gen birth_date_numeric = mdy(birth_month_sort, birth_day_sort, birth_year_sort)
gen birth_month_numeric = ym(birth_year_sort, birth_month_sort)

gen birth_since_baseline_all_v2 = (birth_date_numeric > survey_date_w1) if !missing(survey_date_w1) & !missing(birth_date_numeric)
replace birth_since_baseline_all_v2 = 1 if birth_month_numeric > survey_month_w1 & missing(survey_date_w1) & !missing(survey_month_w1) & !missing(birth_month_numeric)
replace birth_since_baseline_all_v2 = 1 if birth_year_sort > survey_year_w1 & missing(survey_date_w1) & missing(survey_month_w1) & !missing(survey_year_w1) & !missing(birth_year_sort)

gen birth_since_baseline_nosbmc_v2 = (birth_date_numeric > survey_date_w1 & stillbirth != 1 & miscarriage != 1) if !missing(survey_date_w1)  & !missing(birth_date_numeric)
replace birth_since_baseline_nosbmc_v2 = 1 if birth_month_numeric > survey_month_w1 & stillbirth != 1 & miscarriage != 1 & missing(survey_date_w1) & !missing(survey_month_w1) & !missing(birth_month_numeric)
replace birth_since_baseline_nosbmc_v2 = 1 if birth_year_sort > survey_year_w1 & stillbirth != 1 & miscarriage != 1 & missing(survey_date_w1) & missing(survey_month_w1) & !missing(survey_year_w1) & !missing(birth_year_sort)

collapse (sum) births_since_baseline_all_v2=birth_since_baseline_all_v2 births_since_baseline_nosbmc_v2=birth_since_baseline_nosbmc_v2, by(iid survey_round)


//append wave 3 birth records 
append using `births_v2_w3'

tempfile births_v2_w3_w4
save `births_v2_w3_w4', replace 


*merge into master fertility dataset 
use `fertility_allwaves_coded', clear 
merge 1:1 iid survey_round using `births_v2_w3_w4'
tab _merge 
assert _merge != 2
drop _merge 

//save
compress
cd "`PKH'data/coded"
save "mothers_fertility_timing", replace






