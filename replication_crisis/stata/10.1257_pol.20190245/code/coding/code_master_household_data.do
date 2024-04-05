*** 1. INTRODUCTION ***
/*
Description: 	Assembles a panel dataset for households across three survey waves 
				(I, III, and IV) and creates variables for key outcomes and covariates
Uses: 			R, R_AR_01, R_KS1TYPE, R_KS2TYPE, L_FKS1TYPE for each survey wave, 
				kecnames_ben.dta, and pkhben2013_use.dta
Creates: 		household_allwaves_master.dta
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
		global PKH = "$PKH\"	
	}
}

local PKH = "$PKH"
local pkhdata = "`PKH'data/raw/"
local output = "`PKH'output"
local TARGET2 = "$TARGET2"


//Set working directory:
cd "`pkhdata'"

*** 3. APPENDING WAVE III AND IV OBSERVATIONS ***
//First code Wave III survey dates 
use "Wave III/Data/R_KJ_fin.dta", clear 
duplicates report //no pure duplicates 

//use first survey date with at least partial completion 
//drop 13 "third visit" observations that previously had partial completion 
bysort rid (rkj): drop if rkj == 3 & (rhkw[_n-1] == 2 | rhkw[_n-2] == 2) 
//drop 397 "second visit" observations that previously had partial completion 
bysort rid (rkj): drop if rkj == 2 & rhkw[_n-1] == 2
//drop 8 "second visit" observations where respondent refused or was absent but later completed
bysort rid (rkj): drop if rkj == 2 & rhkw == 3
//drop 28 "first visit" observations where respondent refused or was absent but later completed
bysort rid (rkj): drop if rkj == 1 & rhkw == 3

//make sure only one survey date observation per household 
duplicates report rid 

//generate survey date variable 
gen survey_date = mdy(rivw_mm, rivw_dd, rivw_yy)
gen survey_month = ym(rivw_yy, rivw_mm)

//keep only rid and survey_date 
keep rid survey_date survey_month

tempfile surveydates_w3
save `surveydates_w3', replace 

//Load Wave III household data
use "Wave III/Data/R_fin.dta", clear
//drop duplicates 
duplicates drop rid, force 

//merge in survey dates 
merge 1:1 rid using `surveydates_w3' //all but 9 households in master should have a survey date
drop _merge 

//generate 9-digit rid variable
rename rid rid_w3
gen rid = substr(rid_w3, 1, 9)
recast str12 rid

gen split_code = substr(rid_w3, 12, 2)
gen split_indicator = (split_code != "00")

//generate indicator for split variables
gen split_temp = (rar00x == 2)
bysort rid: egen any_split = total(split_temp)
gen split_dummy = (any_split > 0)
drop split_temp any_split 

duplicates report rid 
sort rid

tempfile hh_w3
save `hh_w3'

//First code Wave IV survey dates 
use "Wave IV/R_KJ.dta", clear 
duplicates report //no pure duplicates 

//rkj variable is sometimes coded incorrectly. We need to operate off of chronology of interview visits with at least partial completion
bysort rid (rivw_yy rivw_mm rivw_dd): drop if (rhkw[_n-1] == 2 | rhkw[_n-2] == 2)

//some households refused outright and were never completed later 
duplicates tag rid, gen(dups)
drop if rhkw == 3 & dups != 0
drop dups 

//drop final duplicates 
duplicates tag rid, gen(dups)
drop if rhkw == 2 & dups > 0
drop dups 

//we should now have one observation per household 
//make sure only one survey date observation per household 
duplicates report rid 

//generate survey date variable
destring rivw_mm rivw_dd rivw_yy, replace 
gen survey_date = mdy(rivw_mm, rivw_dd, rivw_yy)
gen survey_month = ym(rivw_yy, rivw_mm)

//keep only rid and survey_date 
keep rid survey_date survey_month

tempfile surveydates_w4
save `surveydates_w4', replace 

//Load Wave IV household data and prepare for append
use "Wave IV/R.dta", clear

//merge in survey dates 
merge 1:1 rid using `surveydates_w4' // all should merge properly 
drop _merge 

//generate 9-digit rid variable
rename rid rid_w4
gen rid = substr(rid_w4, 1, 7) + substr(rid_w4, 10, 2)
recast str12 rid

gen split_code = substr(rid_w4, 14, 4)
gen split_indicator = (split_code != "0000")

//generate indicator for split variables
gen split_temp = (rar00x == 2)
bysort rid: egen any_split = total(split_temp)
gen split_dummy = (any_split > 0)
drop split_temp any_split 


duplicates report rid 
sort rid

tempfile hh_w4
save `hh_w4'

//Load Wave I as master dataset
use "Wave I/Data/R_fin.dta", clear

//no split households at baseline!
gen split_indicator = 0

//Append Wave III and Wave IV
//generate variable, "survey_round", to identify wave of observations 
//(0 = Wave I, 1 = Wave 2, 2 = Wave III)
append using `hh_w3' `hh_w4', generate(survey_round) force
sort rid survey_round


tempfile hh_allwaves
save `hh_allwaves'


*** 4. CODING PKH ELIGIBILITY, RANDOMIZATION, AND TREATMENT STATUS ***
//Open PKH/Kec data 
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
use `hh_allwaves', clear

//generate ea string
gen ea = substr(rid, 1, 3)

//merge with kecamatan treatment list 
merge m:1 ea using `ea_treatment_list' 
tab _merge
drop _merge

//generate dummy for baseline PKH eligibility 
gen elig_pkh_w1 = (survey_round == 0 & (rlk12 == 1 | rlk12 == 2))
//ensure that eligibility dummy is the same across waves for each household 
bysort rid (survey_round): replace elig_pkh_w1=elig_pkh_w1[1]

//generate PKH receipt dummies for HOUSEHOLDS
gen received_pkh_09 = .
replace received_pkh_09 = 0 if survey_round == 1 & !missing(rkr17a)
replace received_pkh_09 = 1 if survey_round == 1 & rkr17a == 1

gen received_pkh_13 = .
replace received_pkh_13 = 0 if survey_round == 2 & !missing(rkr17a)
replace received_pkh_13 = 1 if survey_round == 2 & rkr17a == 1

//generate "received PKH by this survey wave" dummy
gen rid_w3w4 = rid_w3 if survey_round == 1
replace rid_w3w4 = substr(rid_w4, 1, 7) + substr(rid_w4, 10,6) if survey_round == 2
bysort rid_w3w4 (survey_round): carryforward received_pkh_09, gen(pkh_by_this_wave)
replace pkh_by_this_wave = 1 if survey_round == 1 & received_pkh_09 == 1
replace pkh_by_this_wave = 1 if survey_round == 2 & received_pkh_13 == 1
//households added in after baseline 
replace pkh_by_this_wave = 0 if survey_round == 2 & missing(pkh_by_this_wave) & rkr17a == 3

//new endogenous variable: NON-CCT household in CCT kec 
gen non_pkh_hh_cct_kec = (pkh_by_this_wave == 0 & K09 == 1) if survey_round == 1 & !missing(pkh_by_this_wave)
replace non_pkh_hh_cct_kec = (pkh_by_this_wave == 0 & K13 == 1) if survey_round == 2 & !missing(pkh_by_this_wave)

//Drop all household observations that were not randomized at baseline
drop if missing(L07)



//save tempfile
tempfile hh_allwaves 
save `hh_allwaves', replace


*** 5. CODING HOUSEHOLD-LEVEL PER CAPITA CONSUMPTION ***
//Adapted from Generasi do files (village level)

//Wave I coding
//Food expenditures
use "Wave I/Data/R_KS1TYPE_fin.dta", clear
collapse (sum) week_food=rks01, by(rid)
gen month_food = week_food*(30/7)
tempfile food_w1
save `food_w1', replace

//Non-food expenditures
use "Wave I/Data/R_KS2TYPE_fin.dta", clear
//subcategories for education and health 
gen education = rks04 if rks2type == 5
gen health_all = rks04 if (rks2type == 6 | rks2type == 13)
gen health_noinsurance = rks04 if rks2type == 6
//collapse by household
collapse (sum) year_nonfood=rks04 year_education=education ///
	year_health_all=health_all year_health_noinsurance=health_noinsurance, by(rid)
//take monthly values
gen month_nonfood = year_nonfood/12
gen month_education = year_education/12
gen month_health_all = year_health_all/12
gen month_health_noinsurance = year_health_noinsurance/12
gen month_educ_health = month_education + month_health_all
merge 1:1 rid using `food_w1'
tab _merge
keep if _merge==3
drop _merge

//total household expenditure 
gen hhexp = month_food + month_nonfood

tempfile hhexp_w1
save `hhexp_w1', replace

//Merge in household size data
use "Wave I/Data/3_R_AR_01_fin.dta", clear
keep rid rar00
//get rid of pure duplicate observations (15 observations in 2 households)
duplicates drop rid rar00, force 
//generate household size variable 
bysort rid: gen hhsize = _N
bysort rid: keep if _n==1
drop rar00
merge 1:1 rid using `hhexp_w1'
tab _merge
keep if _merge==3
drop _merge

//Indicator for Wave I
gen survey_round = 0

tempfile hhexp_merged_w1
save `hhexp_merged_w1', replace

***

//Wave III coding
use "Wave III/Data/R_fin.dta", clear
duplicates drop rid, force
//Food expenditures
egen week_food = rowtotal(rks01_01-rks01_15), missing
gen month_food = week_food*(30/7)
//Added: milk and egg category
gen week_milk_egg = rks01_05 
gen month_milk_egg = week_milk_egg*(30/7)

keep rid week_food month_food week_milk_egg month_milk_egg
tempfile food_w3
save `food_w3', replace

//Non-food expenditures
use "Wave III/Data/R_KS2TYPE_fin.dta", clear
//subcategories for education and health 
gen education = rks04 if rks2type == 5
gen health_all = rks04 if (rks2type == 6 | rks2type == 13)
gen health_noinsurance = rks04 if rks2type == 6
//collapse by household
collapse (sum) year_nonfood=rks04 year_education=education ///
	year_health_all=health_all year_health_noinsurance=health_noinsurance, by(rid)
//take monthly values
gen month_nonfood = year_nonfood/12
gen month_education = year_education/12
gen month_health_all = year_health_all/12
gen month_health_noinsurance = year_health_noinsurance/12
gen month_educ_health = month_education + month_health_all
merge 1:1 rid using `food_w3'
tab _merge
keep if _merge==3
drop _merge

//total hh expenditure 
gen hhexp = month_food + month_nonfood

tempfile hhexp_w3
save `hhexp_w3', replace

//Merge in household size data
use "Wave III/Data/3_R_AR_01_fin.dta", clear
keep rid rar00 rar01a
//eliminate 13 duplicate observations 
duplicates drop rid rar00, force 

gen lives_in_hh = .
replace lives_in_hh = 1 if (rar01a == 1 | rar01a == 3)
replace lives_in_hh = 0 if (rar01a == 2 | rar01a == 4) 
collapse (sum) hhsize=lives_in_hh, by(rid)
merge 1:1 rid using `hhexp_w3'
tab _merge
keep if _merge==3
drop _merge

//generate 9-digit rid variable
rename rid rid_w3
gen rid = substr(rid_w3, 1, 9)
recast str12 rid

//drop duplicates
sort rid 
duplicates report rid 
*duplicates drop rid, force

//Indicator for Wave III
gen survey_round = 1

tempfile hhexp_merged_w3
save `hhexp_merged_w3', replace

**********

//Wave IV coding
use "Wave IV/R.dta", clear
duplicates drop rid, force

//Food expenditures
egen week_food = rowtotal(rks01_01-rks01_15), missing
gen month_food = week_food*(30/7)
//Added: milk and egg category
gen week_milk_egg = rks01_05 
gen month_milk_egg = week_milk_egg*(30/7)

keep rid week_food month_food week_milk_egg month_milk_egg

tempfile food_w4
save `food_w4', replace

//Non-food expenditures
use "Wave IV/R_KS2TYPE.dta", clear
//subcategories for education and health 
gen education = rks04 if rks2type == 5
gen health_all = rks04 if (rks2type == 6 | rks2type == 13)
gen health_noinsurance = rks04 if rks2type == 6
//collapse by household
collapse (sum) year_nonfood=rks04 year_education=education ///
	year_health_all=health_all year_health_noinsurance=health_noinsurance, by(rid)
//take monthly values
gen month_nonfood = year_nonfood/12
gen month_education = year_education/12
gen month_health_all = year_health_all/12
gen month_health_noinsurance = year_health_noinsurance/12
gen month_educ_health = month_education + month_health_all
merge 1:1 rid using `food_w4'
tab _merge
keep if _merge==3
drop _merge

gen hhexp = month_food + month_nonfood
tempfile hhexp_w4
save `hhexp_w4', replace

//Merge in household size data
use "Wave IV/3_R_AR_01.dta", clear
keep rid rar00 rar00_09 rar01a
gen lives_in_hh = .
replace lives_in_hh = 1 if (rar01a == 1 | rar01a == 3)
replace lives_in_hh = 0 if (rar01a == 2 | rar01a == 4) 
collapse (sum) hhsize=lives_in_hh, by(rid)
merge 1:1 rid using `hhexp_w4'
tab _merge
keep if _merge==3
drop _merge

//generate 9-digit rid variable
rename rid rid_w4
gen rid = substr(rid_w4, 1, 7) + substr(rid_w4, 10, 2)
recast str12 rid

//drop duplicates
duplicates report rid 
*duplicates drop rid, force
sort rid

//Indicator for Wave IV
gen survey_round = 2

tempfile hhexp_merged_w4
save `hhexp_merged_w4', replace

****

//Append expenditure data from all 3 waves into one dataset
use `hhexp_merged_w1', clear
append using `hhexp_merged_w3' `hhexp_merged_w4', force
sort rid survey_round

gen rid_merge = rid if survey_round == 0
replace rid_merge = rid_w3 if survey_round == 1
replace rid_merge = rid_w4 if survey_round == 2

tempfile hh_expenditure_allwaves
save `hh_expenditure_allwaves', replace

****

//now merge expenditure data into full household dataset
use `hh_allwaves', clear
gen rid_merge = rid if survey_round == 0
replace rid_merge = rid_w3 if survey_round == 1
replace rid_merge = rid_w4 if survey_round == 2
merge 1:1 rid_merge survey_round using `hh_expenditure_allwaves' 

keep if _merge == 3
drop _merge

//account for households in Wave III that were missing rar01a
replace hhsize = rhhsize if hhsize == 0 // this gets rid of 367 zero values
sort rid survey_round

//generate per capita expenditure variable
gen pcexp = hhexp/hhsize

tempfile hh_expenditure_merged
save `hh_expenditure_merged', replace

*** 6. CODING VILLAGE- AND HOUSEHOLD-LEVEL COVARIATES ***
//coding availability of health facility in village
*Wave III
use "Wave III/Data/L_FKS1TYPE_fin.dta", clear

keep lid lfks1type lfks02
keep if lfks1type == 1 //only keep observations for puskesmas availability
gen health_facility = (lfks02 == 1)
gen survey_round = 1

drop lfks1type lfks02
tempfile village_health_w3
save `village_health_w3', replace

*Wave IV
use "Wave IV/L_FKS1TYPE.dta", clear

drop if lsplit != "00" 
keep lid lfks1type lfks02
keep if lfks1type == 1 //only keep observations for puskesmas availability
gen health_facility = (lfks02 == 1)
gen survey_round = 2

drop lfks1type lfks02
tempfile village_health_w4
save `village_health_w4', replace

*Wave I and append
use "Wave I/Data/L_FKS1TYPE_fin.dta", clear

keep lid lfks1type lfks02
keep if lfks1type == 1  
gen health_facility = (lfks02 == 1)
gen survey_round = 0

drop lfks1type lfks02
append using `village_health_w3' `village_health_w4', force
sort lid survey_round

tempfile village_health_all
save `village_health_all', replace

*Master household data
use `hh_expenditure_merged', clear 
//generate village identifier for each household
gen lid = substr(rid, 1, 3) + "5" + substr(rid, 5, 3)
//merge 
merge m:1 lid survey_round using `village_health_all'
tab _merge
keep if _merge == 3
drop _merge
sort rid survey_round

*******End of village covariates*******

//Household head education level
tempfile pre_education
save `pre_education'

*Wave III
use "Wave III/Data/3_R_AR_01_fin.dta", clear
keep if rar00 == 1

keep rid rar00 rar11
duplicates drop rid, force 
gen hh_head_education = rar11
drop rar00 rar11
generate survey_round = 1

//generate 9-digit rid variable
rename rid rid_w3
gen rid = substr(rid_w3, 1, 9)
recast str12 rid

//drop duplicates
duplicates report rid 
*duplicates drop rid, force
sort rid

tempfile education_w3
save `education_w3', replace

*Wave IV
use "Wave IV/3_R_AR_01.dta", clear
keep if rar00 == 1 

keep rid rar00 rar11
duplicates drop rid, force
gen hh_head_education = rar11
drop rar00 rar11
generate survey_round = 2

//generate 9-digit rid variable
rename rid rid_w4
gen rid = substr(rid_w4, 1, 7) + substr(rid_w4, 10, 2)
recast str12 rid

//drop duplicates
duplicates report rid 
sort rid

tempfile education_w4
save `education_w4', replace

*Wave I
use "Wave I/Data/3_R_AR_01_fin.dta", clear 
keep if rar00 == 1
keep rid rar00 rar11
duplicates drop rid, force
gen hh_head_education = rar11
drop rar00 rar11 
generate survey_round = 0

append using `education_w3' `education_w4', force
sort rid survey_round

gen rid_merge = rid if survey_round == 0
replace rid_merge = rid_w3 if survey_round == 1
replace rid_merge = rid_w4 if survey_round == 2

tempfile education_allwaves
save `education_allwaves', replace


*merge
use `pre_education', clear 
merge 1:1 rid_merge survey_round using `education_allwaves'

tab _merge
keep if _merge == 3
drop _merge
sort rid survey_round


*******************************************
//Rest of household-level covariates
//dummies
gen hh_head_agr = (rkrt04 == 1) 
gen hh_head_serv = (rkrt04 == 9)
gen clean_water = (rkr04 == 1) 
gen own_latrine = (rkr11 == 1)
gen square_latrine = (rkr12 == 1) 
gen own_septic_tank = (rkr13 == 1) 
gen electricity_PLN = (rkr15 == 1)

//categoricals: create dummies for each category
tab rkr01, gen(roof_type)
	*generates 10 dummies
tab rkr02, gen(wall_type)
	*generates 13 dummies
tab rkr03, gen(floor_type)
	*generates 9 dummies
tab hh_head_education, gen(hh_educ_)
	*generates 10 dummies

//continuous: ln(hhsize) and pcexp
gen hhsize_ln = ln(hhsize)
gen logpcexp = ln(pcexp)

//expenditure categories 
gen pcexp_food = month_food / hhsize
gen pcexp_nonfood = month_nonfood / hhsize 
gen pcexp_education = month_education / hhsize 
gen pcexp_health_all = month_health_all / hhsize
gen pcexp_health_noinsurance = month_health_noinsurance / hhsize
gen pcexp_educ_health = month_educ_health / hhsize
gen pcexp_milk_egg = month_milk_egg / hhsize 

gen logpcexp_food = ln(pcexp_food)
gen logpcexp_nonfood = ln(pcexp_nonfood)
gen logpcexp_education = ln(pcexp_education)
gen logpcexp_health_all = ln(pcexp_health_all)
gen logpcexp_health_noinsurance = ln(pcexp_health_noinsurance)
//categories that have zero values: add 1 inside log 
gen logpcexp_educ_health = ln(pcexp_educ_health + 1)
gen logpcexp_milk_egg = ln(pcexp_milk_egg + 1)

//generate baseline values for controls
sort rid survey_round

//generate dummy to determine if household matches to a baseline observation
bysort rid (survey_round): generate baseline_match = (survey_round[1] == 0)
local characteristics hh_educ_* hh_head_agr hh_head_serv roof_type* ///
	wall_type* floor_type* clean_water own_latrine square_latrine ///
	own_septic_tank electricity_PLN logpcexp logpcexp_food logpcexp_nonfood hhsize_ln 

foreach x of varlist `characteristics' {
	bysort rid (survey_round): gen `x'_baseline = `x'[1]
	replace `x'_baseline = . if baseline_match == 0
}

local characteristics_baseline *baseline 

//identify and account for missing values of baseline characteristics
foreach x of varlist `characteristics_baseline' {
	gen `x'_miss = (`x' == .)
	gen `x'_nm = `x' 
	replace `x'_nm = 0 if `x'_nm == .
}

tempfile hh_coded
save `hh_coded', replace 

*** 7. MERGING WITH KABUPATEN AND KECAMATAN CODES *** 
//merge kabupaten codes in order to FE
use "Wave I/Data/kecnames_ben.dta"
keep lid llk01_cd llk02_cd
rename llk02_cd kabupaten

//provice variable
rename llk01_cd province

tempfile kab
save `kab', replace

use `hh_coded', clear
merge m:1 lid using `kab' 
keep if _merge == 3
drop _merge

//generate numerical kecamatan code 
destring ea, generate(kecamatan)



tempfile hh_allwaves_withlocation
save `hh_allwaves_withlocation', replace


*** 8. CODING OTHER HOUSEHOLD-LEVEL OUTCOME VARIABLES ***
*Alcohol/tobacco cons., land ownership, livestock ownership, HH head employment

****Wave I coding
*Alcohol/tobacco consumption 
use "Wave I/Data/R_KS1TYPE_fin.dta", clear

//drop all consumption categories except alcohol and tobacco
keep if (rks1type == 14 | rks1type == 15)
//sum alcohol and tobacco consumption by HHID for last week 
collapse (sum) week_alcohol_tobacco = rks01, by(rid)
//convert to monthly consumption 
gen month_alcohol_tobacco = week_alcohol_tobacco*(30/7)
*drop week_alcohol_tobacco

tempfile alc_w1
save `alc_w1', replace

*Land ownership (SIZE)
use "Wave I/Data/R_HR1TYPE_fin.dta", clear

//indicator for ownership of each category of land 
gen owns_land_type = (rhr01 == 1)

//create dummy for *any* land ownership
collapse (sum) land_number=owns_land_type total_land_owned_m2=rhr02_1 total_land_owned_ha=rhr02_3, by(rid)
gen owns_any_land = (land_number > 0)
drop land_number

tempfile land_w1
save `land_w1', replace

*Land ownership (TYPES)
use "Wave I/Data/R_HR1TYPE_fin.dta", clear
//reshape data into wide format to separate land types 
keep rid rhr01 rhr1type 
reshape wide rhr01, i(rid) j(rhr1type)
gen owns_irr_rice_field = (rhr011 == 1)
gen owns_rain_rice_field = (rhr012 == 1)
gen owns_dry_land = (rhr013 == 1)
gen owns_land_housing = (rhr014 == 1)
gen owns_land_otherhousing_business = (rhr015 == 1)
keep rid owns*
tempfile land_types_w1
save `land_types_w1', replace


*Livestock/asset ownership
use "Wave I/Data/R_HR2TYPE_fin.dta", clear

//keep only livestock ownership observations (rhr2type 10, 11, 12, 13, 14)
//reshape remaining data into wide format to disaggregate livestock types
reshape wide rhr03 rhr04_n, i(rid) j(rhr2type)
//rename variables appropriately
rename rhr031 		owns_radio
rename rhr04_n1     num_radio 
rename rhr032 		owns_tv
rename rhr04_n2 	num_tv
rename rhr033 		owns_antenna
rename rhr04_n3   	num_antenna
rename rhr034 		owns_showcase
rename rhr04_n4 	num_showcase
rename rhr035 		owns_refrigerator
rename rhr04_n5 	num_refrigerator
rename rhr036 		owns_bicycle
rename rhr04_n6 	num_bicycle
rename rhr037 		owns_motorcycle
rename rhr04_n7 	num_motorcycle
rename rhr038 		owns_car_boat
rename rhr04_n8 	num_car_boat
rename rhr039 		owns_cellphone
rename rhr04_n9 	num_cellphone
rename rhr0310 		owns_chicken
rename rhr04_n10 	num_chicken
rename rhr0311		owns_pig
rename rhr04_n11	num_pig
rename rhr0312		owns_goat
rename rhr04_n12	num_goat
rename rhr0313		owns_cow
rename rhr04_n13	num_cow
rename rhr0314		owns_horse
rename rhr04_n14	num_horse

keep rid owns* num*

//recode no (or "do not know") as 0
foreach x of varlist owns* {
	replace `x' = 0 if (`x' == 3 | `x' == 8)
}

//recode number owned as 0 (instead of missing) if owns_`x' == 0
foreach x in "radio" "tv" "antenna" "showcase" "refrigerator" "bicycle" "motorcycle" "car_boat" ///
				"cellphone" "chicken" "pig" "goat" "cow" "horse" {
	replace num_`x' = 0 if owns_`x' == 0
}

//variable for total number of livestock animals owned 
egen total_livestock_owned = rowtotal(num_chicken num_pig num_goat num_cow num_horse)

tempfile livestock_w1
save `livestock_w1', replace


*Household head employment status last week
use "Wave I/Data/3_R_AR_01_fin.dta", clear

//keep observations only for head of HH
keep if rar00 == 1
//drop 3 duplicate observations 
duplicates drop

//keep only variables of interest
keep rid rar00 rar09 

//generate dummy for head of HH employment, only 1 if "employed", only 0 if "unemployed"
gen head_employed = .
replace head_employed = 1 if rar09 == 1
replace head_employed = 0 if rar09 == 5
drop rar00 rar09 


//merge other outcome variables into a single dataset 
merge 1:1 rid using `alc_w1' 
drop _merge 
merge 1:1 rid using `land_w1' 
drop _merge 
merge 1:1 rid using `land_types_w1' 
drop _merge 
merge 1:1 rid using `livestock_w1'
drop _merge 

//Indicator for Wave I
gen survey_round = 0

tempfile hh_otheroutcomes_w1
save `hh_otheroutcomes_w1', replace


***Wave III
*Alcohol/tobacco consumption 
use "Wave III/Data/R_fin.dta", clear
duplicates drop 
duplicates drop rid, force 

//drop all consumption (and other) variables except alcohol and tobacco
keep rid rks01_14 rks01_15 
//sum alcohol and tobacco consumption by HHID for last week 
egen week_alcohol_tobacco = rowtotal(rks01_14 rks01_15)
//convert to monthly consumption 
gen month_alcohol_tobacco = week_alcohol_tobacco*(30/7)

drop rks01_14 rks01_15

tempfile alc_w3
save `alc_w3', replace

*Land ownership
use "Wave III/Data/R_HR1TYPE_fin.dta", clear
keep rid rhr1type rhr01 rhr02_1 rhr02_3b rhr02_3d

//indicator for ownership of each category of land 
gen owns_land_type = (rhr01 == 1)

//destring hectare-measured area variabes 
replace rhr02_3b = "" if rhr02_3b == "EX" | rhr02_3b == "TT" | rhr02_3b == "TB"
destring rhr02_3b, replace 
//decimal part of total number
replace rhr02_3d = "" if rhr02_3d == "TT"
destring rhr02_3d, replace
replace rhr02_3d = rhr02_3d / 100
gen ha_area = rhr02_3b + rhr02_3d

//create dummy for *any* land ownership
collapse (sum) land_number = owns_land_type total_land_owned_m2=rhr02_1 total_land_owned_ha=ha_area, by(rid)
gen owns_any_land = (land_number > 0)
drop land_number

tempfile land_w3
save `land_w3', replace

*Land ownership (TYPES)
use "Wave III/Data/R_HR1TYPE_fin.dta", clear
//reshape data into wide format to separate land types 
keep rid rhr01 rhr1type 
reshape wide rhr01, i(rid) j(rhr1type)
gen owns_irr_rice_field = (rhr011 == 1)
gen owns_rain_rice_field = (rhr012 == 1)
gen owns_dry_land = (rhr013 == 1)
gen owns_land_housing = (rhr014 == 1)
gen owns_land_otherhousing_business = (rhr015 == 1)
keep rid owns*
tempfile land_types_w3
save `land_types_w3', replace


*Livestock ownership
use "Wave III/Data/R_HR2TYPE_fin.dta", clear

//drop extraneous variables
drop rhr03a rhr04a

//reshape remaining data into wide format to disaggregate livestock types
reshape wide rhr03 rhr04, i(rid) j(rhr2type)
//rename variables appropriately
rename rhr031 		owns_radio
rename rhr041  		num_radio 
rename rhr032 		owns_tv
rename rhr042 		num_tv
rename rhr033 		owns_antenna
rename rhr043   	num_antenna
rename rhr034 		owns_showcase
rename rhr044 		num_showcase
rename rhr035 		owns_refrigerator
rename rhr045 		num_refrigerator
rename rhr036 		owns_bicycle
rename rhr046 		num_bicycle
rename rhr037 		owns_motorcycle
rename rhr047 		num_motorcycle
rename rhr038 		owns_car_boat
rename rhr048 		num_car_boat
rename rhr039 		owns_cellphone
rename rhr049 		num_cellphone
rename rhr0310 		owns_chicken
rename rhr0410 		num_chicken
rename rhr0311		owns_pig
rename rhr0411		num_pig
rename rhr0312		owns_goat
rename rhr0412		num_goat
rename rhr0313		owns_cow
rename rhr0413		num_cow
rename rhr0314		owns_horse
rename rhr0414		num_horse

keep rid owns* num*

//recode no (or "do not know") as 0
foreach x of varlist owns* {
	replace `x' = 0 if (`x' == 3 | `x' == 8)
}

//recode number owned as 0 (instead of missing) if owns_`x' == 0
foreach x in "radio" "tv" "antenna" "showcase" "refrigerator" "bicycle" "motorcycle" "car_boat" ///
				"cellphone" "chicken" "pig" "goat" "cow" "horse" {
	replace num_`x' = 0 if owns_`x' == 0
}

//variable for total number of livestock animals owned 
egen total_livestock_owned = rowtotal(num_chicken num_pig num_goat num_cow num_horse)
tempfile livestock_w3
save `livestock_w3', replace


*Household head employment status last week
use "Wave III/Data/3_R_AR_01_fin.dta", clear

//keep observations only for head of HH
keep if rar02 == 1 //note this is different from Wave I because not all rar00 == 1 are head of household anymore
//drop 3 duplicate observations 
duplicates drop rid, force 

//keep only variables of interest
keep rid rar09_temp

//generate dummy for head of HH employment, only 1 if "employed", only 0 if "unemployed"
gen head_employed = .
replace head_employed = 1 if rar09_temp == 1
replace head_employed = 0 if rar09_temp == 5
drop rar09_temp


//merge other outcome variables into a single dataset 
merge 1:1 rid using `alc_w3' 
drop if _merge != 3
drop _merge 
merge 1:1 rid using `land_w3' 
drop _merge 
merge 1:1 rid using `land_types_w3' 
drop _merge 
merge 1:1 rid using `livestock_w3'
drop _merge 

//generate 9-digit rid variable
rename rid rid_w3
gen rid = substr(rid_w3, 1, 9)
recast str12 rid

duplicates report rid
//Indicator for Wave III
gen survey_round = 1

tempfile hh_otheroutcomes_w3
save `hh_otheroutcomes_w3', replace



**********
***Wave IV
*Alcohol/tobacco consumption 
use "Wave IV/R.dta", clear
duplicates report rid //no duplicates

//drop all consumption (and other) variables except alcohol and tobacco
keep rid rks01_14 rks01_15 
//sum alcohol and tobacco consumption by HHID for last week 
egen week_alcohol_tobacco = rowtotal(rks01_14 rks01_15)
//convert to monthly consumption 
gen month_alcohol_tobacco = week_alcohol_tobacco*(30/7)

drop rks01_14 rks01_15

tempfile alc_w4
save `alc_w4', replace

*Land ownership
use "Wave IV/R_HR1TYPE.dta", clear
keep rid rhr1type rhr01 rhr02_1 rhr02_3

//indicator for ownership of each category of land 
gen owns_land_type = (rhr01 == 1)

//create dummy for *any* land ownership
collapse (sum) land_number=owns_land_type total_land_owned_m2=rhr02_1 total_land_owned_ha=rhr02_3, by(rid)
gen owns_any_land = (land_number > 0)
drop land_number

tempfile land_w4
save `land_w4', replace

*Land ownership (TYPES)
use "Wave IV/R_HR1TYPE.dta", clear
//reshape data into wide format to separate land types 
keep rid rhr01 rhr1type 
reshape wide rhr01, i(rid) j(rhr1type)
gen owns_irr_rice_field = (rhr011 == 1)
gen owns_rain_rice_field = (rhr012 == 1)
gen owns_dry_land = (rhr013 == 1)
gen owns_land_housing = (rhr014 == 1)
gen owns_land_otherhousing_business = (rhr015 == 1)
keep rid owns*
tempfile land_types_w4
save `land_types_w4', replace

*Livestock ownership
use "Wave IV/R_HR2TYPE.dta", clear

//drop extraneous variables
drop rhr03a rhr04a
//reshape remaining data into wide format to disaggregate livestock types
reshape wide rhr03 rhr04, i(rid) j(rhr2type)
//rename variables appropriately
rename rhr031 		owns_radio
rename rhr041  		num_radio 
rename rhr032 		owns_tv
rename rhr042 		num_tv
rename rhr033 		owns_antenna
rename rhr043   	num_antenna
rename rhr034 		owns_showcase
rename rhr044 		num_showcase
rename rhr035 		owns_refrigerator
rename rhr045 		num_refrigerator
rename rhr036 		owns_bicycle
rename rhr046 		num_bicycle
rename rhr037 		owns_motorcycle
rename rhr047 		num_motorcycle
rename rhr038 		owns_car_boat
rename rhr048 		num_car_boat
rename rhr039 		owns_cellphone
rename rhr049 		num_cellphone
rename rhr0310 		owns_chicken
rename rhr0410 		num_chicken
rename rhr0311		owns_pig
rename rhr0411		num_pig
rename rhr0312		owns_goat
rename rhr0412		num_goat
rename rhr0313		owns_cow
rename rhr0413		num_cow
rename rhr0314		owns_horse
rename rhr0414		num_horse

keep rid owns* num*

//recode no (or "do not know") as 0
foreach x of varlist owns* {
	replace `x' = 0 if (`x' == 3 | `x' == 8)
}

//recode number owned as 0 (instead of missing) if owns_`x' == 0
foreach x in "radio" "tv" "antenna" "showcase" "refrigerator" "bicycle" "motorcycle" "car_boat" ///
				"cellphone" "chicken" "pig" "goat" "cow" "horse" {
	replace num_`x' = 0 if owns_`x' == 0
}

//variable for total number of livestock animals owned 
egen total_livestock_owned = rowtotal(num_chicken num_pig num_goat num_cow num_horse)

tempfile livestock_w4
save `livestock_w4', replace


*Household head employment status last week
use "Wave IV/3_R_AR_01.dta", clear

//keep observations only for head of HH
keep if rar02 == 1 //note this is different from Wave I because not all rar00 == 1 are head of household anymore
duplicates report // no duplicates 

//keep only variables of interest
keep rid rar09

//generate dummy for head of HH employment, only 1 if "employed", only 0 if "unemployed"
gen head_employed = .
replace head_employed = 1 if rar09 == 1
replace head_employed = 0 if rar09 == 5
drop rar09


//merge other outcome variables into a single dataset 
merge 1:1 rid using `alc_w4' 
drop _merge 
merge 1:1 rid using `land_w4' 
drop _merge 
merge 1:1 rid using `land_types_w4'
drop _merge 
merge 1:1 rid using `livestock_w4'
drop _merge 
	//no merging issues


//generate 9-digit rid variable
rename rid rid_w4
gen rid = substr(rid_w4, 1, 7) + substr(rid_w4, 10, 2)
recast str12 rid

duplicates report rid

//Indicator for Wave IV
gen survey_round = 2

tempfile hh_otheroutcomes_w4
save `hh_otheroutcomes_w4', replace


//Append expenditure data from all 3 waves into one dataset
use `hh_otheroutcomes_w1', clear
append using `hh_otheroutcomes_w3' `hh_otheroutcomes_w4', force
sort rid survey_round

gen rid_merge = rid if survey_round == 0
replace rid_merge = rid_w3 if survey_round == 1
replace rid_merge = rid_w4 if survey_round == 2

tempfile hh_otheroutcomes_allwaves
save `hh_otheroutcomes_allwaves', replace


**** Number of non-immediate/non-biological children in household ***
*Wave I
//Code survey dates
use "Wave I/Data/R_fin.dta", clear 
duplicates drop 

//first survey date with at least partial completion 
gen survey_date = mdy(rivw_mm_1, rivw_dd_1, rivw_yy_1) if rhkw_1 == 1 | rhkw_1 == 2
replace survey_date = mdy(rivw_mm_2, rivw_dd_2, rivw_yy_2) if rhkw_1 == 3 & rhkw_2 != 3 & missing(survey_date)
replace survey_date = mdy(rivw_mm_3, rivw_dd_3, rivw_yy_3) if rhkw_1 == 3 & rhkw_2 == 3 & missing(survey_date)
replace survey_date = mdy(rivw_mm_1, rivw_dd_1, rivw_yy_1) if missing(survey_date)

keep rid survey_date
tempfile survey_date_w1
save `survey_date_w1', replace 

//Household roster
use "Wave I/Data/3_R_AR_01_fin.dta", clear 
duplicates drop 

merge m:1 rid using `survey_date_w1'
tab _merge  
keep if _merge != 2 // all should merge 
drop _merge 

//"other relatives" < 18 years old 
gen non_immediate_child = (rar02 == 9 & age_years < 18)
//adopted children 
replace non_immediate_child = 1 if rar02 == 11
//anyone else under 18 where both parents do not live in hh
gen dad_dead_notin_hh = (rar05 == 51 | rar05 == 52)
gen mom_dead_notin_hh = (rar06 == 51 | rar05 == 52) 
replace non_immediate_child = 1 if age_years < 18 & dad_dead_notin_hh == 1 & mom_dead_notin_hh == 1

//collapse by household 
collapse (sum) num_non_immediate_child = non_immediate_child, by(rid)
gen survey_round = 0

tempfile non_immediate_child_w1
save `non_immediate_child_w1', replace 




*Wave III
//Wave III survey dates 
use "Wave III/Data/R_KJ_fin.dta", clear 
duplicates report //no pure duplicates 

//use first survey date with at least partial completion 
//drop 13 "third visit" observations that previously had partial completion 
bysort rid (rkj): drop if rkj == 3 & (rhkw[_n-1] == 2 | rhkw[_n-2] == 2) 
//drop 397 "second visit" observations that previously had partial completion 
bysort rid (rkj): drop if rkj == 2 & rhkw[_n-1] == 2
//drop 8 "second visit" observations where respondent refused or was absent but later completed
bysort rid (rkj): drop if rkj == 2 & rhkw == 3
//drop 28 "first visit" observations where respondent refused or was absent but later completed
bysort rid (rkj): drop if rkj == 1 & rhkw == 3

//make sure only one survey date observation per household 
duplicates report rid 

//generate survey date variable 
gen survey_date = mdy(rivw_mm, rivw_dd, rivw_yy)

//keep only rid and survey_date 
keep rid survey_date 

tempfile surveydate_w3
save `surveydate_w3', replace 

//Load Wave III household data
use "Wave III/Data/3_R_AR_01_fin.dta", clear
duplicates drop 

merge m:1 rid using `surveydate_w3'
tab _merge  
keep if _merge != 2 // all should merge 
drop _merge 


//"other relatives" < 18 years old 
gen non_immediate_child = (rar02 == 9 & age_years < 18)
//adopted children 
replace non_immediate_child = 1 if rar02 == 11
//anyone else under 18 where both parents do not live in hh
gen dad_dead_notin_hh = (rar05 == 51 | rar05 == 52)
gen mom_dead_notin_hh = (rar06 == 51 | rar05 == 52) 
replace non_immediate_child = 1 if age_years < 18 & dad_dead_notin_hh == 1 & mom_dead_notin_hh == 1

//collapse by household 
collapse (sum) num_non_immediate_child = non_immediate_child, by(rid)
gen survey_round = 1

tempfile non_immediate_child_w3
save `non_immediate_child_w3', replace 


*Wave IV
//First code Wave IV survey dates 
use "Wave IV/R_KJ.dta", clear 
duplicates report //no pure duplicates 

//rkj variable is sometimes coded incorrectly. We need to operate off of chronology of interview visits with at least partial completion
bysort rid (rivw_yy rivw_mm rivw_dd): drop if (rhkw[_n-1] == 2 | rhkw[_n-2] == 2)

//some households refused outright and were never completed later 
duplicates tag rid, gen(dups)
drop if rhkw == 3 & dups != 0
drop dups 

//drop final duplicates 
duplicates tag rid, gen(dups)
drop if rhkw == 2 & dups > 0
drop dups 

//we should now have one observation per household 
//make sure only one survey date observation per household 
duplicates report rid 

//generate survey date variable
destring rivw_mm rivw_dd rivw_yy, replace 
gen survey_date = mdy(rivw_mm, rivw_dd, rivw_yy)

//keep only rid and survey_date 
keep rid survey_date

tempfile surveydate_w4
save `surveydate_w4', replace 


//Wave IV Roster 
use "Wave IV/3_R_AR_01.dta", clear 
duplicates drop 

merge m:1 rid using `surveydate_w4'
tab _merge  
keep if _merge != 2 // all should merge 
drop _merge 

//"other relatives" < 18 years old 
gen non_immediate_child = (rar02 == 9 & age_years < 18)
//adopted children 
replace non_immediate_child = 1 if rar02 == 11
//anyone else under 18 where both parents do not live in hh
gen dad_dead_notin_hh = (rar05 == 51 | rar05 == 52)
gen mom_dead_notin_hh = (rar06 == 51 | rar05 == 52) 
replace non_immediate_child = 1 if age_years < 18 & dad_dead_notin_hh == 1 & mom_dead_notin_hh == 1

//collapse by household 
collapse (sum) num_non_immediate_child = non_immediate_child, by(rid)
gen survey_round = 2


//append Waves I and III
append using `non_immediate_child_w1' `non_immediate_child_w3', force
sort rid survey_round

rename rid rid_merge 


tempfile hh_non_immediate_child_allwaves
save `hh_non_immediate_child_allwaves', replace


****

**** Monetary contribution to community service activities ***
*Wave I
use "Wave I/Data/R_PM14_fin.dta", clear 
duplicates drop 

//drop empty lines 
drop if missing(rpm14type)

//collapse both types of activities but keep donation types (cash vs. in-kind) separate 
collapse (sum) rpm17_a rpm17_b, by(rid)
gen total_donations = rpm17_a + rpm17_b

gen survey_round = 0

tempfile donations_w1
save `donations_w1', replace 


*Wave III
use "Wave III/Data/R_PM14_fin.dta", clear 
duplicates drop 

//collapse both types of activities but keep donation types separate 
collapse (sum) rpm17_a rpm17_b, by(rid)
gen total_donations = rpm17_a + rpm17_b

gen survey_round = 1

tempfile donations_w3
save `donations_w3', replace 


*Wave IV 
use "Wave IV/R_PM14.dta", clear 
duplicates drop 

//collapse both types of activities but keep donation types separate 
collapse (sum) rpm17_a rpm17_b, by(rid)
gen total_donations = rpm17_a + rpm17_b

gen survey_round = 2


//append all waves and save 
append using `donations_w1' `donations_w3', force 

sort rid survey_round
rename rid rid_merge 

tempfile hh_donations_allwaves
save `hh_donations_allwaves', replace 

****


//now merge expenditure data into full household dataset
use `hh_allwaves_withlocation', clear
merge 1:1 rid_merge survey_round using `hh_otheroutcomes_allwaves'

keep if _merge == 3
drop _merge


//merge non-biological children 
merge 1:1 rid_merge survey_round using `hh_non_immediate_child_allwaves' 

keep if _merge == 3
drop _merge 

//merge HH donations 
merge 1:1 rid_merge survey_round using `hh_donations_allwaves' 

replace rpm17_a = 0 if _merge == 1
replace rpm17_b = 0 if _merge == 1
replace total_donations = 0 if _merge == 1
drop if _merge == 2
drop _merge 

//variable for log per-capita donations 
gen total_donations_percapita = total_donations/hhsize 
gen cash_donations_percapita = rpm17_a/hhsize 
gen inkind_donations_percapita = rpm17_b/hhsize 

gen lnpc_total_donations = ln(total_donations_percapita + 1)
gen lnpc_cash_donations = ln(cash_donations_percapita + 1)
gen lnpc_inkind_donations = ln(inkind_donations_percapita + 1)

//create variable for pcexp on alcohol and tobacco
gen pcexp_alctobacco = month_alcohol_tobacco/hhsize 
gen logpcexp_alctobacco = ln(pcexp_alctobacco + 1)


*** MERGING IN VILLAGE-LEVEL BASELINE SUPPLY-SIDE COVARIATES *** 
gen lid_merge = substr(rid, 1, 3) + "5" + substr(rid, 5, 3) if survey_round == 0
replace lid_merge = substr(rid, 1, 3) + "5" + substr(rid, 5, 3) if survey_round == 1
replace lid_merge = substr(rid_w4, 1, 3) + "5" + substr(rid_w4, 5, 5) if survey_round == 2

cd "`PKH'data/coded"
merge m:1 lid_merge survey_round using "village_outcomes_master", ///
			keepusing(lack_health_fac_top3 lack_med_equip_top3 lack_hlthworkers_top3 low_hlth_aware_top3 lack_sch_fac_top3 lack_sch_infr_top3 lack_sch_teachers_top3 lack_sch_aware_top3 ///
					 lack_health_fac_top3_bl lack_med_equip_top3_bl lack_hlthworkers_top3_bl low_hlth_aware_top3_bl lack_sch_fac_top3_bl lack_sch_infr_top3_bl lack_sch_teachers_top3_bl lack_sch_aware_top3_bl ///
					 num_totaldoc_live_pc num_totaldoc_prac_pc num_nurse_live_pc num_nurse_prac_pc num_totalmidwife_live_pc num_totalmidwife_prac_pc num_tradbirth_live_pc num_tradbirth_prac_pc total_primary_pc total_secondary_pc ///
					 num_totaldoc_live_pc_bl num_totaldoc_prac_pc_bl num_nurse_live_pc_bl num_nurse_prac_pc_bl num_totalmidwife_live_pc_bl num_totalmidwife_prac_pc_bl num_tradbirth_live_pc_bl num_tradbirth_prac_pc_bl total_primary_pc_bl total_secondary_pc_bl)
tab _merge 
drop if _merge == 2
drop _merge 

***CODING DATES OF HOUSEHOLD AND KABUPATEN PKH RECEIPT***
//code the month in which household last received PKH 
gen hh_pkh_month = ym(rkr17c_yy, rkr17c_mm)
//take kabupaten-level average PKH receipt month 
bysort kabupaten survey_round: egen kab_avg_pkh_month = mean(hh_pkh_month)
//variable for difference between survey month and last PKH receipt month
gen survey_month_diff = survey_month - kab_avg_pkh_month
summ survey_month_diff, detail 
gen survey_1month_kab_pkh = (survey_month_diff <= 1) //the HH was surveyed within 1 month of the kab-average PKH receipt 
gen survey_2month_kab_pkh = (survey_month_diff <= 2)
gen survey_3month_kab_pkh = (survey_month_diff <= 3)
gen survey_4month_kab_pkh = (survey_month_diff <= 4)

***CODE INDICATOR FOR WHETHER HOUSEHOLD WAS SURVEYED AT ENDLINE***
sort rid survey_round 
gen endline_dummy = (survey_round == 2)
bysort rid_w3w4: egen hh_surveyed_endline = total(endline_dummy) if survey_round != 0
replace hh_surveyed_endline = 1 if hh_surveyed_endline > 0 & !missing(hh_surveyed_endline)


//save
compress
cd "`PKH'data/coded"
save "household_allwaves_master", replace





