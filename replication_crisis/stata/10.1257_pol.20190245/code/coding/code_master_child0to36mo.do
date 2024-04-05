
*** 1. INTRODUCTION ***
/*
Description: 	Coding master panel dataset for children 0-36 months
Uses: 			Raw survey data + WHO "igrowup" module for anthropometrics
Creates: 		child0to36_allwaves_master.dta and output from WHO module
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

cd "`PKH'"

//Set working directory:
cd "`pkhdata'"


*** 3. ASSEMBLING IMMUNIZATION DATA BY SURVEY WAVE ***
*WAVE I*
//Load Wave I immunization schedule data 
use "Wave I/Data/T_IM1TYPE_fin", clear 

//Drop duplicates and keep only variables of interest
duplicates drop //1 duplicate observation
keep tid tim1type tim07 tim09 tim09_dd tim09_mm tim09_yy

//Reshape data into wide format (one observation per child)
reshape wide tim07 tim09 tim09_dd tim09_mm tim09_yy, i(tid) j(tim1type)
//Drop variables for "other" immunization categories
drop tim0713 tim0714 tim0913 tim09_dd13 tim09_mm13 tim09_yy13 tim0914 tim09_dd14 tim09_mm14 tim09_yy14

//Rename immunization variables
rename tim071 bcg
rename tim072 polio_1
rename tim073 polio_2
rename tim074 polio_3
rename tim075 polio_4
rename tim076 dpt_1
rename tim077 dpt_2
rename tim078 dpt_3
rename tim079 measles
rename tim0710 hepb_1
rename tim0711 hepb_2
rename tim0712 hepb_3

//Recode "no" from 3 to 0 and "don't know" to missing 
foreach x of var bcg polio_1 polio_2 polio_3 polio_4 dpt_1 dpt_2 dpt_3 measles hepb_1 hepb_2 hepb_3{
	recode `x' (3 = 0) (8 = 0)
}

//Count total number of immunizations received for each child 
egen num_imm_receive = rowtotal(bcg polio_1 polio_2 polio_3 polio_4 dpt_1 dpt_2 dpt_3 measles hepb_1 hepb_2 hepb_3), missing

//Create dates of each vaccine
gen date_bcg 	 = mdy(tim09_mm1, tim09_dd1, tim09_yy1)
gen date_polio_1 = mdy(tim09_mm2, tim09_dd2, tim09_yy2)
gen date_polio_2 = mdy(tim09_mm3, tim09_dd3, tim09_yy3)
gen date_polio_3 = mdy(tim09_mm4, tim09_dd4, tim09_yy4)
gen date_polio_4 = mdy(tim09_mm5, tim09_dd5, tim09_yy5)
gen date_dpt_1 	 = mdy(tim09_mm6, tim09_dd6, tim09_yy6)
gen date_dpt_2 	 = mdy(tim09_mm7, tim09_dd7, tim09_yy7)
gen date_dpt_3 	 = mdy(tim09_mm8, tim09_dd8, tim09_yy8)
gen date_measles = mdy(tim09_mm9, tim09_dd9, tim09_yy9)
gen date_hepb_1  = mdy(tim09_mm10, tim09_dd10, tim09_yy10)
gen date_hepb_2	 = mdy(tim09_mm11, tim09_dd11, tim09_yy11)
gen date_hepb_3  = mdy(tim09_mm12, tim09_dd12, tim09_yy12)

//Drop original date variables 
drop tim09*

sort tid
tempfile imm_w1
save `imm_w1', replace

//Open main infant data file 
use "Wave I/Data/T_fin", clear 
duplicates report //no duplicate observations

//Generate survey dates using first survey attempt with at least partial completion
gen survey_date = mdy(tivw_mm_1, tivw_dd_1, tivw_yy_1) if thkw_1 == 1 | thkw_1 == 2
replace survey_date = mdy(tivw_mm_2, tivw_dd_2, tivw_yy_2) if survey_date== . & thkw_2==1 | thkw_2==2
replace survey_date = mdy(tivw_mm_3, tivw_dd_3, tivw_yy_3) if survey_date== . & thkw_3==1 | thkw_3==2

//merge with immunization record
merge 1:1 tid using `imm_w1'
tab _merge 
drop _merge

//generate breastfeeding indicators
//whether child has ever been breastfed 
gen ever_breastfed = .
replace ever_breastfed = 0 if tna01 == 3
replace ever_breastfed = 1 if tna01 == 1

//whether child was breastfed within 1 hour of birth
gen breastfed_1hr = 0 if tna01 == 3
replace breastfed_1hr = 1 if tna02_1 < 60
replace breastfed_1hr = 1 if tna02_2 == 1
replace breastfed_1hr = 0 if tna02_2 > 1 & tna02_2 < .
replace breastfed_1hr = 0 if tna02_3 > 0 & tna02_3 < .

//whether child was breastfed exclusively for first 3 months of life
gen excl_breastfed_3mon = 0 if tna01 == 3
replace excl_breastfed_3mon = 1 if tna03 == 6
replace excl_breastfed_3mon = 1 if tna03_n >= 3 & tna03_n < .
replace excl_breastfed_3mon = . if tna05_u == 8 | tna06_u == 8
replace excl_breastfed_3mon = . if tna03 == 8
replace excl_breastfed_3mon = 0 if tna03_n <= 2
replace excl_breastfed_3mon = 0 if tna05_u == 1 | (tna05_u == 2 & tna05_n <= 13) | (tna05_u == 3 & tna05_n <= 3)
replace excl_breastfed_3mon = 0 if tna06_u == 1 | (tna06_u == 2 & tna06_n <= 13) | (tna06_u == 3 & tna06_n <= 3)

//Generate variable for complete immunization schedule given age
generate imm_age_complete=.
replace imm_age_complete=1 if age>=7 & age<30.4375 & hepb_1==1
replace imm_age_complete=0 if age>=7 & age<30.4375 & (hepb_1!=1 & !missing(hepb_1))
replace imm_age_complete=1 if age>=30.4375 & age<60.875 & hepb_1==1 & bcg==1
replace imm_age_complete=0 if age>=30.4375 & age<60.875 & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)))
replace imm_age_complete=1 if age>=60.875 & age<91.3125 & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1
replace imm_age_complete=0 if age>=60.875 & age<91.3125 & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)))
replace imm_age_complete=1 if age>=91.3125 & age<121.75 & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1
replace imm_age_complete=0 if age>=91.3125 & age<121.75 & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)))
replace imm_age_complete=1 if age>=121.75 & age<273.9375 & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1 & dpt_3==1 & polio_3==1
replace imm_age_complete=0 if age>=121.75 & age<273.9375 & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)) | (dpt_3!=1 & !missing(dpt_3)) | (polio_3!=1 & !missing(polio_3)))
replace imm_age_complete=1 if age>=273.9375 & age<. & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1 & dpt_3==1 & polio_3==1 & measles==1 & polio_4==1
replace imm_age_complete=0 if age>=273.9375 & age<. & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)) | (dpt_3!=1 & !missing(dpt_3)) | (polio_3!=1 & !missing(polio_3)) | (measles!=1 & !missing(measles)) | (polio_4!=1 & !missing(polio_4)))

//Generate variable calculating the total number of immunizations received, restricting to those supposed to have received at that age
egen num_imm_receive_for_age_1 = rowtotal(hepb_1), missing 
egen num_imm_receive_for_age_2 = rowtotal(hepb_1 bcg), missing
egen num_imm_receive_for_age_3 = rowtotal(hepb_1 bcg hepb_2 dpt_1 polio_1), missing
egen num_imm_receive_for_age_4 = rowtotal(hepb_1 bcg hepb_2 dpt_1 polio_1 hepb_3 dpt_2 polio_2), missing
egen num_imm_receive_for_age_5 = rowtotal(hepb_1 bcg hepb_2 dpt_1 polio_1 hepb_3 dpt_2 polio_2 dpt_3 polio_3), missing 
egen num_imm_receive_for_age_6 = rowtotal(hepb_1 bcg hepb_2 dpt_1 polio_1 hepb_3 dpt_2 polio_2 dpt_3 polio_3 measles polio_4), missing

gen num_imm_receive_for_age = .
replace num_imm_receive_for_age = num_imm_receive_for_age_1 if age>=7 & age<30.4375
replace num_imm_receive_for_age = num_imm_receive_for_age_2 if age>=30.4375 & age<60.875
replace num_imm_receive_for_age = num_imm_receive_for_age_3 if age>=60.875 & age<91.3125
replace num_imm_receive_for_age = num_imm_receive_for_age_4 if age>=91.3125 & age<121.75
replace num_imm_receive_for_age = num_imm_receive_for_age_5 if age>=121.75 & age<273.9375
replace num_imm_receive_for_age = num_imm_receive_for_age_6 if age>=273.9375 & age<.

//drop placeholder variables 
forvalues x = 1/6 {
	drop num_imm_receive_for_age_`x'
}

//Count should be zero if child is not yet 7 days old OR if missing all immunizations 
replace num_imm_receive_for_age = . if num_imm_receive == .

//hard-code number of required immunizations by age 
gen imm_age_number_req = .
replace imm_age_number_req = 1  if age>=7        & age<30.4375 
replace imm_age_number_req = 2  if age>=30.4375  & age<60.875 
replace imm_age_number_req = 5  if age>=60.875   & age<91.3125 
replace imm_age_number_req = 8  if age>=91.3125  & age<121.75 
replace imm_age_number_req = 10 if age>=121.75   & age<273.9375 
replace imm_age_number_req = 12  if age>=273.9375 & age<. 

//Generate variable for percent of required immunizations received by age (for children up to 11 months of age and 23 months old and below)
gen imm_age_uptak_percent_all = num_imm_receive / imm_age_number_req
gen imm_age_uptak_diff_all = num_imm_receive - imm_age_number_req

gen imm_age_uptak_percent_only = num_imm_receive_for_age / imm_age_number_req
gen imm_age_uptak_diff_only = num_imm_receive_for_age - imm_age_number_req

gen imm_uptak_pct_23mons_all = imm_age_uptak_percent_all if age < 700.0625
gen imm_uptak_diff_23mons_all = imm_age_uptak_diff_all if age < 700.0625

gen imm_uptak_pct_23mons = imm_age_uptak_percent_only if age < 700.0625
gen imm_uptak_diff_23mons = imm_age_uptak_diff_only if age < 700.0625

//Generate variable for total immunization
generate imm_total_complete=.
replace imm_total_complete=1 if hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1 & dpt_3==1 & polio_3==1 & measles==1 & polio_4==1
replace imm_total_complete=0 if ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)) | (dpt_3!=1 & !missing(dpt_3)) | (polio_3!=1 & !missing(polio_3)) | (measles!=1 & !missing(measles)) | (polio_4!=1 & !missing(polio_4)))
replace imm_total_complete=. if age < 7 | missing(age)

//Generate variable for total immunization in children at least 10 months of age
generate imm_total_comp10month = .
replace imm_total_comp10month = 1 if age >= 304.375 & age < . & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1 & dpt_3==1 & polio_3==1 & measles==1 & polio_4==1
replace imm_total_comp10month = 0 if age >= 304.375 & age < . & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)) | (dpt_3!=1 & !missing(dpt_3)) | (polio_3!=1 & !missing(polio_3)) | (measles!=1 & !missing(measles)) | (polio_4!=1 & !missing(polio_4)))


//also create indicators for only children 0-5 years of age 
foreach x of varlist imm_age_complete imm_total_complete imm_total_comp10month imm_age_uptak_percent_only {
	gen yr5_`x' = `x' if age <= 1826.25
	gen yr3_`x' = `x' if age <= 1095.75
}


//code dummies for health monitoring / mother-child health book 
gen has_kms = (tim01 == 1) if !missing(tim01)
gen has_buku_kia = (tim03 == 1) if !missing(tim03)
gen showed_card_records = (tim05 == 1 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)
gen showed_card_no_records = (tim05 == 2 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)
gen card_not_shown = (tim05 == 3 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)  
gen card_lost_other = (tim05 > 3 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)

tempfile child_w1
save `child_w1', replace 


*WAVE III*
//Load Wave III immunization schedule data 
use "Wave III/Data/T_IM1TYPE_fin", clear 

//Drop duplicates and keep only variables of interest
duplicates report // no duplicate observations
keep tid tim1type tim07 tim09 tim09_dd tim09_mm tim09_yy

//Reshape data into wide format (one observation per child)
reshape wide tim07 tim09 tim09_dd tim09_mm tim09_yy, i(tid) j(tim1type)
//Drop variables for "other" immunization categories and HepB 4 (wasn't in Wave I)
drop tim0713 tim0714 tim0715 *13 *14 *15

//Rename immunization variables
rename tim071 bcg
rename tim072 polio_1
rename tim073 polio_2
rename tim074 polio_3
rename tim075 polio_4
rename tim076 dpt_1
rename tim077 dpt_2
rename tim078 dpt_3
rename tim079 measles
rename tim0710 hepb_1
rename tim0711 hepb_2
rename tim0712 hepb_3

//Recode "no" from 3 to 0 and "don't know" to missing 
foreach x of var bcg polio_1 polio_2 polio_3 polio_4 dpt_1 dpt_2 dpt_3 measles hepb_1 hepb_2 hepb_3{
	recode `x' (3 = 0) (8 = 0)
}

//Count total number of immunizations received for each child 
egen num_imm_receive = rowtotal(bcg polio_1 polio_2 polio_3 polio_4 dpt_1 dpt_2 dpt_3 measles hepb_1 hepb_2 hepb_3), missing


//Create dates of each vaccine
gen date_bcg 	 = mdy(tim09_mm1, tim09_dd1, tim09_yy1)
gen date_polio_1 = mdy(tim09_mm2, tim09_dd2, tim09_yy2)
gen date_polio_2 = mdy(tim09_mm3, tim09_dd3, tim09_yy3)
gen date_polio_3 = mdy(tim09_mm4, tim09_dd4, tim09_yy4)
gen date_polio_4 = mdy(tim09_mm5, tim09_dd5, tim09_yy5)
gen date_dpt_1 	 = mdy(tim09_mm6, tim09_dd6, tim09_yy6)
gen date_dpt_2 	 = mdy(tim09_mm7, tim09_dd7, tim09_yy7)
gen date_dpt_3 	 = mdy(tim09_mm8, tim09_dd8, tim09_yy8)
gen date_measles = mdy(tim09_mm9, tim09_dd9, tim09_yy9)
gen date_hepb_1  = mdy(tim09_mm10, tim09_dd10, tim09_yy10)
gen date_hepb_2	 = mdy(tim09_mm11, tim09_dd11, tim09_yy11)
gen date_hepb_3  = mdy(tim09_mm12, tim09_dd12, tim09_yy12)

//Drop original date variables 
drop tim09*

sort tid
tempfile imm_w3
save `imm_w3', replace

//Open survey date file 
use "Wave III/Data/T_KJ_fin", clear 

//Want to use first survey date with at least partial completion, and only 1 survey date per child
//drops 2 "third visit" observations that previously had partial completion 
bysort tid (tkj): drop if tkj == 3 & (thkw[_n-1] == 2 | thkw[_n-2] == 2) 
//drops 103 "second visit" observations that previously had partial completion 
bysort tid (tkj): drop if tkj == 2 & thkw[_n-1] == 2
//drops 7 "first visit" observations where respondent refused or was absent but later completed
bysort tid (tkj): drop if tkj == 1 & thkw == 3

duplicates report tid //ensure only 1 survey date obsevation per child 

sort tid 
tempfile dates_w3
save `dates_w3', replace 

//Code HOUSEHOLD survey dates for merge with roster info 
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
//note: labeling as "alt" because only used for people who are ONLY in household roster 
gen survey_date_alt = mdy(rivw_mm, rivw_dd, rivw_yy)
gen survey_month_alt = ym(rivw_yy, rivw_mm)

//keep only rid and survey_date 
keep rid survey_date_alt survey_month_alt
tempfile hh_surveydates_w3
save `hh_surveydates_w3', replace 

//now code household roster info 
//Merge survey dates with household roster then append observations from individuals who moved out of HH 
use "Wave III/Data/R_AR_01_fin.dta", clear
duplicates drop 

//merge in survey dates 
merge m:1 rid using `hh_surveydates_w3' //all merge successfully
drop _merge

//only need to keep kids that will merge in (because only kids 0-60 were weighed)
keep if age_years_alt < 9

//rename gender variable so matches what we will be using for calculating z-scores
rename rar03 tir02

//generate tid variable for use in merging 
//convert child's HH serial number into string in order to create tid for matching
tostring rar00, replace format(%02.0f) 
gen tid = substr(rid,1,3) + "4" + substr(rid,5,9) + rar00

//identify any duplicates (luckily, there are 0)
duplicates report tid 
duplicates tag tid, gen(tag)
drop if tag == 1
drop tag

keep tid age_alt age_years_alt tir02 imputed_birth_day month_of_birth missing_birth_month missing_birth_date

tempfile roster_info_w3
save `roster_info_w3', replace 



//Open household data file for infant vitals--for weight and height in WHO calcs 
use "Wave III/Data/R_US_fin", clear

//destring mother weight variable 
destring rus03_ab rus03_ad, replace force

//code decimal values of weight appropriately
replace rus03_ad = rus03_ad /100

//generate maternal weight variable 
gen mom_weight = rus03_ab
replace mom_weight = rus03_ab + rus03_ad if rus03_ad != .
bysort rid rus01a: egen mother_weight = mean(mom_weight)
drop mom_weight

//only use average measurements
keep if rus_row == 4

//destring weight and height variables
destring rus03_cb rus03_cd rus01_b rus01_d, replace force

//code decimal values of weight and height appropriately
replace rus03_cd = rus03_cd /100
replace rus01_d = rus01_d /100

//generate weight variable 
gen weight = rus03_cb
replace weight = rus03_cb + rus03_cd if rus03_cd != .

//generate height variable
gen height = rus01_b
replace height = rus01_b + rus01_d if rus01_d != .

//convert child's HH serial number into string in order to create tid for matching
tostring rus01a, replace format(%02.0f) 
gen tid = substr(rid,1,3) + "4" + substr(rid,5,9) + rus01a


//Drop 8 observations containing measurement discrepancies for 4 children 
duplicates tag tid, gen(tag)
drop if tag == 1

keep tid height weight mother_weight
duplicates drop
sort tid

//merge in roster info 
//note that any observations that have _merge == 2 are simply kids in roster who were not weighed 
merge 1:1 tid using `roster_info_w3'
tab _merge
tab _merge age_years_alt
drop if _merge == 2
drop _merge 

replace missing_birth_month = 1 if missing(missing_birth_month)
replace missing_birth_date = 1 if missing(missing_birth_date)

tempfile heightweight_w3
save `heightweight_w3', replace 


//code nutritional consumption
use "Wave III/Data/T_NATYPE_fin.dta", clear 
duplicates report //no pure duplicates 

//reshape into wide format so one observation for each child 
reshape wide tna08 tna09 tna09_n, i(tid) j(tnatype)

//recode missing day counts as 0 except for few "do not know"s 
forvalues i = 1/14 {
	replace tna09_n`i' = 0 if missing(tna09_n`i') & tna08`i' == 3
	drop tna09`i' //can get rid of tna09 variable (labeling number of days)
	recode tna08`i' (3 = 0) //recode "Tidak" from 3 to 0
}

//rename variables accordingly
rename tna081		ate_milk
rename tna09_n1		days_ate_milk
rename tna082		ate_egg
rename tna09_n2		days_ate_egg
rename tna083		ate_beef
rename tna09_n3		days_ate_beef
rename tna084		ate_pork
rename tna09_n4		days_ate_pork
rename tna085		ate_chicken_duck
rename tna09_n5		days_ate_chicken_duck
rename tna086		ate_fish
rename tna09_n6		days_ate_fish
rename tna087		ate_rice
rename tna09_n7		days_ate_rice
rename tna088		ate_other_grain
rename tna09_n8		days_ate_other_grain
rename tna089		ate_tubers
rename tna09_n9		days_ate_tubers
rename tna0810		ate_veg
rename tna09_n10	days_ate_veg
rename tna0811		ate_fruit
rename tna09_n11	days_ate_fruit
rename tna0812		ate_inst_noodle
rename tna09_n12	days_ate_inst_noodle
rename tna0813		ate_snack
rename tna09_n13	days_ate_snack
rename tna0814		ate_sweets
rename tna09_n14	days_ate_sweets

tempfile nutrition_consumption_w3
save `nutrition_consumption_w3', replace 

//Open main infant data file 
use "Wave III/Data/T_fin", clear 
duplicates drop // 6 pure duplicate observations
duplicates drop tid, force // 1 more duplicate observation with a coding typo

//merge in survey date info 
merge 1:1 tid using `dates_w3'
tab _merge
drop _merge

//Generate survey dates
gen survey_date = mdy(tivw_mm, tivw_dd, tivw_yy)

//merge with immunization record
merge 1:1 tid using `imm_w3'
tab _merge
drop _merge

//generate breastfeeding indicators
//whether child has ever been breastfed 
gen ever_breastfed = .
replace ever_breastfed = 0 if tna01 == 3
replace ever_breastfed = 1 if tna01 == 1

//whether child was breastfed within 1 hour of birth
gen breastfed_1hr = 0 if tna01 == 3
replace breastfed_1hr = 1 if tna02_1 < 60
replace breastfed_1hr = 1 if tna02_2 == 1
replace breastfed_1hr = 0 if tna02_2 > 1 & tna02_2 < .
replace breastfed_1hr = 0 if tna02_3 > 0 & tna02_3 < .

//whether child was breastfed exclusively for first 3 months of life
gen excl_breastfed_3mon = 0 if tna01 == 3
replace excl_breastfed_3mon = 1 if tna03 == 6
replace excl_breastfed_3mon = 1 if tna03_n >= 3 & tna03_n < .
replace excl_breastfed_3mon = . if tna05_u == 8 | tna06_u == 8
replace excl_breastfed_3mon = . if tna03 == 8
replace excl_breastfed_3mon = 0 if tna03_n <= 2
replace excl_breastfed_3mon = 0 if tna05_u == 1 | (tna05_u == 2 & tna05_n <= 13) | (tna05_u == 3 & tna05_n <= 3)
replace excl_breastfed_3mon = 0 if tna06_u == 1 | (tna06_u == 2 & tna06_n <= 13) | (tna06_u == 3 & tna06_n <= 3)

//Generate variable for complete immunization schedule given age
generate imm_age_complete=.
replace imm_age_complete=1 if age>=7 & age<30.4375 & hepb_1==1
replace imm_age_complete=0 if age>=7 & age<30.4375 & (hepb_1!=1 & !missing(hepb_1))
replace imm_age_complete=1 if age>=30.4375 & age<60.875 & hepb_1==1 & bcg==1
replace imm_age_complete=0 if age>=30.4375 & age<60.875 & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)))
replace imm_age_complete=1 if age>=60.875 & age<91.3125 & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1
replace imm_age_complete=0 if age>=60.875 & age<91.3125 & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)))
replace imm_age_complete=1 if age>=91.3125 & age<121.75 & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1
replace imm_age_complete=0 if age>=91.3125 & age<121.75 & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)))
replace imm_age_complete=1 if age>=121.75 & age<273.9375 & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1 & dpt_3==1 & polio_3==1
replace imm_age_complete=0 if age>=121.75 & age<273.9375 & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)) | (dpt_3!=1 & !missing(dpt_3)) | (polio_3!=1 & !missing(polio_3)))
replace imm_age_complete=1 if age>=273.9375 & age<. & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1 & dpt_3==1 & polio_3==1 & measles==1 & polio_4==1
replace imm_age_complete=0 if age>=273.9375 & age<. & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)) | (dpt_3!=1 & !missing(dpt_3)) | (polio_3!=1 & !missing(polio_3)) | (measles!=1 & !missing(measles)) | (polio_4!=1 & !missing(polio_4)))

//Generate variable calculating the total number of immunizations received, restricting to those supposed to have received at that age
egen num_imm_receive_for_age_1 = rowtotal(hepb_1), missing 
egen num_imm_receive_for_age_2 = rowtotal(hepb_1 bcg), missing
egen num_imm_receive_for_age_3 = rowtotal(hepb_1 bcg hepb_2 dpt_1 polio_1), missing
egen num_imm_receive_for_age_4 = rowtotal(hepb_1 bcg hepb_2 dpt_1 polio_1 hepb_3 dpt_2 polio_2), missing
egen num_imm_receive_for_age_5 = rowtotal(hepb_1 bcg hepb_2 dpt_1 polio_1 hepb_3 dpt_2 polio_2 dpt_3 polio_3), missing 
egen num_imm_receive_for_age_6 = rowtotal(hepb_1 bcg hepb_2 dpt_1 polio_1 hepb_3 dpt_2 polio_2 dpt_3 polio_3 measles polio_4), missing

gen num_imm_receive_for_age = .
replace num_imm_receive_for_age = num_imm_receive_for_age_1 if age>=7 & age<30.4375
replace num_imm_receive_for_age = num_imm_receive_for_age_2 if age>=30.4375 & age<60.875
replace num_imm_receive_for_age = num_imm_receive_for_age_3 if age>=60.875 & age<91.3125
replace num_imm_receive_for_age = num_imm_receive_for_age_4 if age>=91.3125 & age<121.75
replace num_imm_receive_for_age = num_imm_receive_for_age_5 if age>=121.75 & age<273.9375
replace num_imm_receive_for_age = num_imm_receive_for_age_6 if age>=273.9375 & age<.

//drop placeholder variables 
forvalues x = 1/6 {
	drop num_imm_receive_for_age_`x'
}

//Count should be zero if child is not yet 7 days old OR if missing all immunizations 
replace num_imm_receive_for_age = . if num_imm_receive == .

//hard-code number of required immunizations by age 
gen imm_age_number_req = .
replace imm_age_number_req = 1  if age>=7        & age<30.4375 
replace imm_age_number_req = 2  if age>=30.4375  & age<60.875 
replace imm_age_number_req = 5  if age>=60.875   & age<91.3125 
replace imm_age_number_req = 8  if age>=91.3125  & age<121.75 
replace imm_age_number_req = 10 if age>=121.75   & age<273.9375 
replace imm_age_number_req = 12  if age>=273.9375 & age<. 

//Generate variable for percent of required immunizations received by age (for children up to 11 months of age and 23 months old and below)
gen imm_age_uptak_percent_all = num_imm_receive / imm_age_number_req
gen imm_age_uptak_diff_all = num_imm_receive - imm_age_number_req

gen imm_age_uptak_percent_only = num_imm_receive_for_age / imm_age_number_req
gen imm_age_uptak_diff_only = num_imm_receive_for_age - imm_age_number_req

gen imm_uptak_pct_23mons_all = imm_age_uptak_percent_all if age < 700.0625
gen imm_uptak_diff_23mons_all = imm_age_uptak_diff_all if age < 700.0625

gen imm_uptak_pct_23mons = imm_age_uptak_percent_only if age < 700.0625
gen imm_uptak_diff_23mons = imm_age_uptak_diff_only if age < 700.0625

//Generate variable for total immunization
generate imm_total_complete=.
replace imm_total_complete=1 if hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1 & dpt_3==1 & polio_3==1 & measles==1 & polio_4==1
replace imm_total_complete=0 if ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)) | (dpt_3!=1 & !missing(dpt_3)) | (polio_3!=1 & !missing(polio_3)) | (measles!=1 & !missing(measles)) | (polio_4!=1 & !missing(polio_4)))
replace imm_total_complete=. if age < 7 | missing(age)

//Generate variable for total immunization in children at least 10 months of age
generate imm_total_comp10month = .
replace imm_total_comp10month = 1 if age >= 304.375 & age < . & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1 & dpt_3==1 & polio_3==1 & measles==1 & polio_4==1
replace imm_total_comp10month = 0 if age >= 304.375 & age < . & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)) | (dpt_3!=1 & !missing(dpt_3)) | (polio_3!=1 & !missing(polio_3)) | (measles!=1 & !missing(measles)) | (polio_4!=1 & !missing(polio_4)))

//also create indicators for only children 0-5 years of age 
foreach x of varlist imm_age_complete imm_total_complete imm_total_comp10month imm_age_uptak_percent_only {
	gen yr5_`x' = `x' if age <= 1826.25
	gen yr3_`x' = `x' if age <= 1095.75
}


//merge with height and weight data from household file 
merge 1:1 tid using `heightweight_w3'
tab _merge

//do not drop observations that were only in "using" HH weight/height file, because we can still use their gender and coded age
//now, incorporate age info from alternate roster file, where available 
replace age = age_alt if missing(age) & _merge == 2
replace age_years = age_years_alt if missing(age_years) & _merge == 2
drop _merge


//merge with nutrition consumption data 
merge 1:1 tid using `nutrition_consumption_w3'
tab _merge //make sure none are using only, but do not drop non-merges
drop _merge


//code dummies for health monitoring / mother-child health book 
gen has_kms = (tim01 == 1) if !missing(tim01)
gen has_buku_kia = (tim03 == 1) if !missing(tim03)
gen showed_card_records = (tim05 == 1 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)
gen showed_card_no_records = (tim05 == 2 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)
gen card_not_shown = (tim05 == 3 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)
gen card_kept_relatives = (tim05 == 4 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)
gen card_kept_office = (tim05 == 5 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)
gen card_lost_other = (tim05 > 5 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)

//create 11-digit tid variable for matching with Wave I
rename tid tid_w3
gen tid = substr(tid_w3, 1, 9) + substr(tid_w3, 14, 2)

tempfile child_w3
save `child_w3', replace 



*WAVE IV*
//Load Wave IV immunization schedule data 
use "Wave IV/T_IM1TYPE", clear 

//Drop duplicates and keep only variables of interest
duplicates report // no duplicate observations
keep tid tim1type tim07 tim09 tim09_dd tim09_mm tim09_yy

//destring date variables
destring tim09_dd tim09_mm tim09_yy, replace
//Reshape data into wide format (one observation per child)
reshape wide tim07 tim09 tim09_dd tim09_mm tim09_yy, i(tid) j(tim1type)
//Drop variables for "other" immunization categories and HepB 4 (wasn't in Wave I)
drop tim0713 tim0714 tim0715 *13 *14 *15 *17 *18 *95 *98

//Rename immunization variables
rename tim071 bcg
rename tim072 polio_1
rename tim073 polio_2
rename tim074 polio_3
rename tim075 polio_4
rename tim076 dpt_1
rename tim077 dpt_2
rename tim078 dpt_3
rename tim079 measles
rename tim0710 hepb_1
rename tim0711 hepb_2
rename tim0712 hepb_3

//Recode "no" from 3 to 0 and "don't know" to missing 
foreach x of var bcg polio_1 polio_2 polio_3 polio_4 dpt_1 dpt_2 dpt_3 measles hepb_1 hepb_2 hepb_3{
	recode `x' (3 = 0) (8 = 0)
}

//Count total number of immunizations received for each child 
egen num_imm_receive = rowtotal(bcg polio_1 polio_2 polio_3 polio_4 dpt_1 dpt_2 dpt_3 measles hepb_1 hepb_2 hepb_3), missing

//Create dates of each vaccine
gen date_bcg 	 = mdy(tim09_mm1, tim09_dd1, tim09_yy1)
gen date_polio_1 = mdy(tim09_mm2, tim09_dd2, tim09_yy2)
gen date_polio_2 = mdy(tim09_mm3, tim09_dd3, tim09_yy3)
gen date_polio_3 = mdy(tim09_mm4, tim09_dd4, tim09_yy4)
gen date_polio_4 = mdy(tim09_mm5, tim09_dd5, tim09_yy5)
gen date_dpt_1 	 = mdy(tim09_mm6, tim09_dd6, tim09_yy6)
gen date_dpt_2 	 = mdy(tim09_mm7, tim09_dd7, tim09_yy7)
gen date_dpt_3 	 = mdy(tim09_mm8, tim09_dd8, tim09_yy8)
gen date_measles = mdy(tim09_mm9, tim09_dd9, tim09_yy9)
gen date_hepb_1  = mdy(tim09_mm10, tim09_dd10, tim09_yy10)
gen date_hepb_2	 = mdy(tim09_mm11, tim09_dd11, tim09_yy11)
gen date_hepb_3  = mdy(tim09_mm12, tim09_dd12, tim09_yy12)

//Drop original date variables 
drop tim09*

sort tid
tempfile imm_w4
save `imm_w4', replace

//Open survey date file 
use "Wave IV/T_KJ", clear 

//Want to use first survey date with at least partial completion, and only 1 survey date per child
count if tkj == 3 //0 "third visit" observations
//drops 74 "second visit" observations that previously had partial completion 
bysort tid (tkj): drop if tkj == 2 & (thkw[_n-1] == 2 | thkw[_n-1] == 1)
	//there is one tid for which tkj == 2 is coded as happening BEFORE the second interview, when tkj == 1
//drops 3 "first visit" observations where respondent refused or was absent but later completed
bysort tid (tkj): drop if tkj == 1 & thkw == 3

duplicates report tid //ensure only 1 survey date obsevation per child 

//destring date variables
destring tivw_dd tivw_mm tivw_yy, replace

sort tid 
tempfile dates_w4
save `dates_w4', replace 



//Code HOUSEHOLD survey dates for merge with roster info 
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
//note: labeling as "alt" used for people who are ONLY in household roster 
destring rivw_mm rivw_dd rivw_yy, replace 
gen survey_date_alt = mdy(rivw_mm, rivw_dd, rivw_yy)
gen survey_month_alt = ym(rivw_yy, rivw_mm)

//keep only rid and survey_date 
keep rid survey_date_alt survey_month_alt

tempfile hh_surveydates_w4
save `hh_surveydates_w4', replace 

//now code household roster info 
//Merge survey dates with household roster
use "Wave IV/R_AR_01.dta", clear
duplicates drop 

//merge in survey dates 
merge m:1 rid using `hh_surveydates_w4' //all merge successfully
drop _merge


//keep kids we know will merge in
keep if age_years_alt < 9

//rename gender variable so matches what we will be using to calculate z-scores
rename rar03 tir02

//generate tid variable for use in merging 
//convert child's HH serial number into string in order to create tid for matching
tostring rar00, replace format(%02.0f) 
gen tid = substr(rid,1,3) + "4" + substr(rid,5,13) + rar00

//drop any duplicates
duplicates report tid 
duplicates tag tid, gen(tag)
drop if tag == 1
drop tag

keep tid age_alt age_years_alt tir02 imputed_birth_day month_of_birth missing_birth_month missing_birth_date

tempfile roster_info_w4
save `roster_info_w4', replace 

//Open household data file for infant vitals--for weight and height in WHO calcs 
use "Wave IV/US_IV_updated", clear

//generate mother_weight
bysort rid rus01a_1: egen mother_weight = mean(rus03_a_1)


//only use average measurements
keep if rus_row_1 == 4

//generate weight variable 
gen weight = rus03_c_1

//generate height variable
gen height = rus01

//convert child's HH serial number into string in order to create tid for matching
tostring rus01a_1, replace format(%02.0f) 
gen tid = substr(rid,1,3) + "4" + substr(rid,5,13) + rus01a_1


//identify any duplicates (luckily, there are 0)
duplicates tag tid, gen(tag)
drop if tag == 1

keep tid height weight mother_weight 
duplicates drop
sort tid

//merge in roster info 
//note that any observations that have _merge == 2 are simply kids in roster who were not weighed 
merge 1:1 tid using `roster_info_w4'
tab _merge
tab _merge age_years_alt
drop if _merge == 2
drop _merge 

//rename tid variable for merging with main child file (since these do not always correspond 1:1)
rename tid tid_hh_merge

replace missing_birth_month = 1 if missing(missing_birth_month)
replace missing_birth_date = 1 if missing(missing_birth_date)

tempfile heightweight_w4
save `heightweight_w4', replace 

//code nutritional consumption
use "Wave IV/T_NATYPE.dta", clear 
duplicates report //no pure duplicates 
drop if tnatype == 16 // drop "other" consumption category (most are "no" regardless)

//reshape into wide format so one observation for each child 
reshape wide tna08 tna09 tna09_n, i(tid) j(tnatype)

//recode missing day counts as 0 except for few "do not know"s 
forvalues i = 1/15 {
	replace tna09_n`i' = 0 if missing(tna09_n`i') & tna08`i' == 3
	drop tna09`i' //can get rid of tna09 variable (labeling number of days)
	recode tna08`i' (3 = 0) //recode "Tidak" from 3 to 0
}

//rename variables accordingly
rename tna081		ate_milk
rename tna09_n1		days_ate_milk
rename tna082		ate_egg
rename tna09_n2		days_ate_egg
rename tna083		ate_beef
rename tna09_n3		days_ate_beef
rename tna084		ate_pork
rename tna09_n4		days_ate_pork
rename tna085		ate_chicken_duck
rename tna09_n5		days_ate_chicken_duck
rename tna086		ate_fish
rename tna09_n6		days_ate_fish
rename tna087		ate_rice
rename tna09_n7		days_ate_rice
rename tna088		ate_other_grain
rename tna09_n8		days_ate_other_grain
rename tna089		ate_tubers
rename tna09_n9		days_ate_tubers
rename tna0810		ate_veg
rename tna09_n10	days_ate_veg
rename tna0811		ate_fruit
rename tna09_n11	days_ate_fruit
rename tna0812		ate_inst_noodle
rename tna09_n12	days_ate_inst_noodle
rename tna0813		ate_snack
rename tna09_n13	days_ate_snack
rename tna0814		ate_sweets
rename tna09_n14	days_ate_sweets
rename tna0815		ate_processed
rename tna09_n15	days_ate_processed

tempfile nutrition_consumption_w4
save `nutrition_consumption_w4', replace 


//Open main infant data file 
use "Wave IV/T", clear 
duplicates report //no duplicate observations!

//merge in survey date info 
merge 1:1 tid using `dates_w4'
tab _merge // all should have merged successfully
drop _merge

//Generate survey dates
gen survey_date = mdy(tivw_mm, tivw_dd, tivw_yy)

//merge with immunization record
merge 1:1 tid using `imm_w4'
tab _merge
drop _merge

//generate breastfeeding indicators
//whether child has ever been breastfed 
gen ever_breastfed = .
replace ever_breastfed = 0 if tna01 == 3
replace ever_breastfed = 1 if tna01 == 1

//whether child was breastfed within 1 hour of birth
gen breastfed_1hr = 0 if tna01 == 3
replace breastfed_1hr = 1 if tna02_1 < 60
replace breastfed_1hr = 1 if tna02_2 == 1
replace breastfed_1hr = 0 if tna02_2 > 1 & tna02_2 < .
replace breastfed_1hr = 0 if tna02_3 > 0 & tna02_3 < .

//whether child was breastfed exclusively for first 3 months of life
gen excl_breastfed_3mon = 0 if tna01 == 3
replace excl_breastfed_3mon = 1 if tna03 == 6
replace excl_breastfed_3mon = 1 if tna03_n >= 3 & tna03_n < .
replace excl_breastfed_3mon = . if tna05_u == 8 | tna06_u == 8
replace excl_breastfed_3mon = . if tna03 == 8
replace excl_breastfed_3mon = 0 if tna03_n <= 2
replace excl_breastfed_3mon = 0 if tna05_u == 1 | (tna05_u == 2 & tna05_n <= 13) | (tna05_u == 3 & tna05_n <= 3)
replace excl_breastfed_3mon = 0 if tna06_u == 1 | (tna06_u == 2 & tna06_n <= 13) | (tna06_u == 3 & tna06_n <= 3)

//Generate variable for complete immunization schedule given age
generate imm_age_complete=.
replace imm_age_complete=1 if age>=7 & age<30.4375 & hepb_1==1
replace imm_age_complete=0 if age>=7 & age<30.4375 & (hepb_1!=1 & !missing(hepb_1))
replace imm_age_complete=1 if age>=30.4375 & age<60.875 & hepb_1==1 & bcg==1
replace imm_age_complete=0 if age>=30.4375 & age<60.875 & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)))
replace imm_age_complete=1 if age>=60.875 & age<91.3125 & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1
replace imm_age_complete=0 if age>=60.875 & age<91.3125 & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)))
replace imm_age_complete=1 if age>=91.3125 & age<121.75 & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1
replace imm_age_complete=0 if age>=91.3125 & age<121.75 & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)))
replace imm_age_complete=1 if age>=121.75 & age<273.9375 & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1 & dpt_3==1 & polio_3==1
replace imm_age_complete=0 if age>=121.75 & age<273.9375 & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)) | (dpt_3!=1 & !missing(dpt_3)) | (polio_3!=1 & !missing(polio_3)))
replace imm_age_complete=1 if age>=273.9375 & age<. & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1 & dpt_3==1 & polio_3==1 & measles==1 & polio_4==1
replace imm_age_complete=0 if age>=273.9375 & age<. & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)) | (dpt_3!=1 & !missing(dpt_3)) | (polio_3!=1 & !missing(polio_3)) | (measles!=1 & !missing(measles)) | (polio_4!=1 & !missing(polio_4)))

//Generate variable calculating the total number of immunizations received, restricting to those supposed to have received at that age
egen num_imm_receive_for_age_1 = rowtotal(hepb_1), missing 
egen num_imm_receive_for_age_2 = rowtotal(hepb_1 bcg), missing
egen num_imm_receive_for_age_3 = rowtotal(hepb_1 bcg hepb_2 dpt_1 polio_1), missing
egen num_imm_receive_for_age_4 = rowtotal(hepb_1 bcg hepb_2 dpt_1 polio_1 hepb_3 dpt_2 polio_2), missing
egen num_imm_receive_for_age_5 = rowtotal(hepb_1 bcg hepb_2 dpt_1 polio_1 hepb_3 dpt_2 polio_2 dpt_3 polio_3), missing 
egen num_imm_receive_for_age_6 = rowtotal(hepb_1 bcg hepb_2 dpt_1 polio_1 hepb_3 dpt_2 polio_2 dpt_3 polio_3 measles polio_4), missing

gen num_imm_receive_for_age = .
replace num_imm_receive_for_age = num_imm_receive_for_age_1 if age>=7 & age<30.4375
replace num_imm_receive_for_age = num_imm_receive_for_age_2 if age>=30.4375 & age<60.875
replace num_imm_receive_for_age = num_imm_receive_for_age_3 if age>=60.875 & age<91.3125
replace num_imm_receive_for_age = num_imm_receive_for_age_4 if age>=91.3125 & age<121.75
replace num_imm_receive_for_age = num_imm_receive_for_age_5 if age>=121.75 & age<273.9375
replace num_imm_receive_for_age = num_imm_receive_for_age_6 if age>=273.9375 & age<.

//drop placeholder variables 
forvalues x = 1/6 {
	drop num_imm_receive_for_age_`x'
}

//Count should be zero if child is not yet 7 days old OR if missing all immunizations 
replace num_imm_receive_for_age = . if num_imm_receive == .


//hard-code number of required immunizations by age 
gen imm_age_number_req = .
replace imm_age_number_req = 1  if age>=7        & age<30.4375 
replace imm_age_number_req = 2  if age>=30.4375  & age<60.875 
replace imm_age_number_req = 5  if age>=60.875   & age<91.3125 
replace imm_age_number_req = 8  if age>=91.3125  & age<121.75 
replace imm_age_number_req = 10 if age>=121.75   & age<273.9375 
replace imm_age_number_req = 12  if age>=273.9375 & age<. 

//Generate variable for percent of required immunizations received by age (for children up to 11 months of age and 23 months old and below)
gen imm_age_uptak_percent_all = num_imm_receive / imm_age_number_req
gen imm_age_uptak_diff_all = num_imm_receive - imm_age_number_req

gen imm_age_uptak_percent_only = num_imm_receive_for_age / imm_age_number_req
gen imm_age_uptak_diff_only = num_imm_receive_for_age - imm_age_number_req

gen imm_uptak_pct_23mons_all = imm_age_uptak_percent_all if age < 700.0625
gen imm_uptak_diff_23mons_all = imm_age_uptak_diff_all if age < 700.0625

gen imm_uptak_pct_23mons = imm_age_uptak_percent_only if age < 700.0625
gen imm_uptak_diff_23mons = imm_age_uptak_diff_only if age < 700.0625

//Generate variable for total immunization
generate imm_total_complete=.
replace imm_total_complete=1 if hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1 & dpt_3==1 & polio_3==1 & measles==1 & polio_4==1
replace imm_total_complete=0 if ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)) | (dpt_3!=1 & !missing(dpt_3)) | (polio_3!=1 & !missing(polio_3)) | (measles!=1 & !missing(measles)) | (polio_4!=1 & !missing(polio_4)))
replace imm_total_complete=. if age < 7 | missing(age)

//Generate variable for total immunization in children at least 10 months of age
generate imm_total_comp10month = .
replace imm_total_comp10month = 1 if age >= 304.375 & age < . & hepb_1==1 & bcg==1 & hepb_2==1 & dpt_1==1 & polio_1==1 & hepb_3==1 & dpt_2==1 & polio_2==1 & dpt_3==1 & polio_3==1 & measles==1 & polio_4==1
replace imm_total_comp10month = 0 if age >= 304.375 & age < . & ((hepb_1!=1 & !missing(hepb_1)) | (bcg!=1 & !missing(bcg)) | (hepb_2!=1 & !missing(hepb_2)) | (dpt_1!=1 & !missing(dpt_1)) | (polio_1!=1 & !missing(dpt_1)) | (hepb_3!=1 & !missing(hepb_3)) | (dpt_2!=1 & !missing(dpt_2)) | (polio_2!=1 & !missing(polio_2)) | (dpt_3!=1 & !missing(dpt_3)) | (polio_3!=1 & !missing(polio_3)) | (measles!=1 & !missing(measles)) | (polio_4!=1 & !missing(polio_4)))

//also create indicators for only children 0-5 years of age 
foreach x of varlist imm_age_complete imm_total_complete imm_total_comp10month imm_age_uptak_percent_only {
	gen yr5_`x' = `x' if age <= 1826.25
	gen yr3_`x' = `x' if age <= 1095.75
}

//merge with nutrition consumption data 
merge 1:1 tid using `nutrition_consumption_w4'
tab _merge //make sure none are using only, but do not drop non-merges
drop _merge

//fix tid of kids who have improper tid code to match with proper rid 
tostring tir01_cd, replace format(%02.0f) 
gen indicator_temp = substr(tid, 12, 2)
gen tid_hh_merge = substr(tid, 1, 17) + tir01_cd 
replace tid_hh_merge = substr(tid, 1, 11) + "00" + substr(tid, 14, 4) + tir01_cd if indicator_temp == "13"
drop indicator_temp

//merge with height and weight data from household file 
merge 1:1 tid_hh_merge using `heightweight_w4'
tab _merge
//do not drop observations that were only in "using" HH weight/height file, because we can still use their gender and coded age
//now, incorporate age info from alternate roster file, where available 
replace age = age_alt if missing(age) & _merge == 2
replace age_years = age_years_alt if missing(age_years) & _merge == 2
drop _merge

//destring posyandu visit variable, which will be used later for coding number of weighings in last 3 months 
replace tpos05 = "" if tpos05 == "TT"
destring tpos05, replace 

//code dummies for health monitoring / mother-child health book 
gen has_kms = (tim01 == 1) if !missing(tim01)
gen has_buku_kia = (tim03 == 1) if !missing(tim03)
gen showed_card_records = (tim05 == 1 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)
gen showed_card_no_records = (tim05 == 2 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)
gen card_not_shown = (tim05 == 5 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05) //note different value coding between waves III and IV 
gen card_kept_relatives = (tim05 == 3 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05) //note different value coding between waves III and IV 
gen card_kept_office = (tim05 == 4 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05) //note different value coding between waves III and IV 
gen card_lost_other = (tim05 > 5 & (has_kms == 1 | has_buku_kia == 1)) if !missing(tim05)


//create 11-digit tid variable to match with Wave I observations
rename tid tid_w4
gen tid = substr(tid_w4, 1, 7) + substr(tid_w4, 10, 2) + substr(tid_w4, 18, 2)

tempfile child_w4
save `child_w4', replace 


*** 4. APPENDING OBSERVATIONS FROM ALL WAVES ***
use `child_w1', clear 

//rename weight and height variables to match variable names in III and IV 
//this used to be in Section 5 but needs to be done before append 
rename tus03_c weight
rename tus01 height

//append Waves III and IV and generate survey round variable 
append using `child_w3' `child_w4', generate(survey_round) force 
sort tid survey_round 


*** 5. CODING NUTRITION OUTCOMES USING WHO FILES ***
//take care of any negative ages caused by improperly reported birthday 
replace age = . if age < 0
replace age_years = . if age_years < 0

//Rename variables for use in WHO command
rename tir02 gender

//recode female as 2 rather than 3
replace gender = 2 if gender == 3

//code height/length measure for Wave I (WHO file will impute for Waves III and IV)
replace tus02=. if tus02==7
gen str1 measure="l" if tus02==1
replace measure="h" if tus02==2
replace measure="." if tus02==.

//save final data file
cd "`PKH'data/coded"
save "child0to36months_allwaves_prenutrition", replace


//begin WHO macro
clear
set more 1

adopath + "`PKH'WHO igrowup STATA/"

//load dataset 
use "child0to36months_allwaves_prenutrition", clear 


//Generate first three parameters for WHO command
*gen str90 reflib= "`PKH'WHO igrowup STATA/"
gen str180 reflib= "`PKH'WHO igrowup STATA/"
lab var reflib "Directory of reference tables"

*gen str90 datalib="`PKH'data/coded/"
gen str180 datalib="`PKH'data/coded/"
lab var datalib "Directory for datafiles"

gen str40 datalab="child0to36months_allwaves_nutrition"
lab var datalab "Working file"

//Define age unit
gen str6 ageunit="days"

//Define oedema variable
gen str1 oedema="n"

//Define sampling weight variable
gen sw=1

//Run WHO command
igrowup_restricted reflib datalib datalab gender age ageunit weight height measure oedema sw

//end WHO macro
clear

*** 6. CODING CHILD NUTRITION OUTCOMES ***
//use new datafile with nutrition info
use "child0to36months_allwaves_nutrition_z_rc.dta", clear 

//underweight (for age)
gen mal_weightforage=.
replace mal_weightforage=0 if _zwei>-2 & _zwei<.
replace mal_weightforage=1 if _zwei<=-2

//severe underweight (for age)
gen severe_weightforage=.
replace severe_weightforage=0 if _zwei>-3 & _zwei<.
replace severe_weightforage=1 if _zwei<=-3

//stunting
gen mal_heightforage=.
replace mal_heightforage=0 if _zlen>-2 & _zlen<.
replace mal_heightforage=1 if _zlen<=-2

//severe stunting
gen severe_heightforage=.
replace severe_heightforage=0 if _zlen>-3 & _zlen<.
replace severe_heightforage=1 if _zlen<=-3

//wasting
gen mal_weightforheight=.
replace mal_weightforheight=0 if _zwfl>-2 & _zwfl<.
replace mal_weightforheight=1 if _zwfl<=-2

//severe wasting
gen severe_weightforheight=.
replace severe_weightforheight=0 if _zwfl>-3 & _zwfl<.
replace severe_weightforheight=1 if _zwfl<=-3

//do stunting outcomes by age 
gen age_months = floor(age / 30.4375)

gen stunted_neonatal = .
replace stunted_neonatal = 0 if mal_heightforage == 0 & age <=28
replace stunted_neonatal = 1 if mal_heightforage == 1 & age <=28

gen stunted_1to12months = .
replace stunted_1to12months = 0 if mal_heightforage == 0 & age > 28 & age_months < 12
replace stunted_1to12months = 1 if mal_heightforage == 1 & age > 28 & age_months < 12

gen stunted_1to2years = .
replace stunted_1to2years = 0 if mal_heightforage == 0 & age_years >= 1 & age_years < 2
replace stunted_1to2years = 1 if mal_heightforage == 1 & age_years >= 1 & age_years < 2

gen stunted_2to3years = .
replace stunted_2to3years = 0 if mal_heightforage == 0 & age_years == 2
replace stunted_2to3years = 1 if mal_heightforage == 1 & age_years == 2

gen stunted_3to4years = .
replace stunted_3to4years = 0 if mal_heightforage == 0 & age_years == 3
replace stunted_3to4years = 1 if mal_heightforage == 1 & age_years == 3

gen stunted_4to5years = .
replace stunted_4to5years = 0 if mal_heightforage == 0 & age_years == 4
replace stunted_4to5years = 1 if mal_heightforage == 1 & age_years == 4

gen severe_stunted_neonatal = .
replace severe_stunted_neonatal = 0 if severe_heightforage == 0 & age <= 28
replace severe_stunted_neonatal = 1 if severe_heightforage == 1 & age <= 28

gen severe_stunted_1to12months = .
replace severe_stunted_1to12months = 0 if severe_heightforage == 0 & age > 28 & age_months < 12
replace severe_stunted_1to12months = 1 if severe_heightforage == 1 & age > 28 & age_months < 12

gen severe_stunted_1to2years = .
replace severe_stunted_1to2years = 0 if severe_heightforage == 0 & age_years >= 1 & age_years < 2
replace severe_stunted_1to2years = 1 if severe_heightforage == 1 & age_years >= 1 & age_years < 2

gen severe_stunted_2to3years = .
replace severe_stunted_2to3years = 0 if severe_heightforage == 0 & age_years == 2
replace severe_stunted_2to3years = 1 if severe_heightforage == 1 & age_years == 2

gen severe_stunted_3to4years = .
replace severe_stunted_3to4years = 0 if severe_heightforage == 0 & age_years == 3
replace severe_stunted_3to4years = 1 if severe_heightforage == 1 & age_years == 3

gen severe_stunted_4to5years = .
replace severe_stunted_4to5years = 0 if severe_heightforage == 0 & age_years == 4
replace severe_stunted_4to5years = 1 if severe_heightforage == 1 & age_years == 4

//Other child morbidity outcomes  
gen diarrhea_lastmonth = (tmaa01 == 1)
replace diarrhea_lastmonth = . if (tmaa01 == . | tmaa01 == 8)

gen fever_lastmonth = (tmaa07 == 1)
replace fever_lastmonth = . if (tmaa07 == . | tmaa07 == 8)

gen cough_lastmonth = (tmaa08 == 1)
replace cough_lastmonth = . if (tmaa08 == . | tmaa08 == 8)

gen fevercough_lastmonth = (fever_lastmonth == 1 | cough_lastmonth == 1)
replace fevercough_lastmonth = . if missing(fever_lastmonth) & missing(cough_lastmonth)

//create age-restricted version of the above indicators
gen diarrhea_lastmonth_0to5 = diarrhea_lastmonth if age_months <= 60
gen fevercough_lastmonth_0to5 = fevercough_lastmonth if age_months <= 60 

//Generate total times weighed in last 3 months (use mom's recall of # posyandu visits in last 3 months (POS05) but 0 if was not weighed at last visit)
gen times_weighed_last3months = tpos05
replace times_weighed_last3months = 0 if tpos01 == 3
replace times_weighed_last3months = 0 if tpos07a == 3

//don't want to consider "0"s from children older than 5 years 
gen times_weighed_0to5 = times_weighed_last3months if age_months <= 60

//Generate total number of times Vitamin A taken
gen vitA_total = tim06_1
replace vitA_total = tim06_2 if tim06_1 == .

gen vitA_total_6mons_2years = vitA_total if age >= 182.625 & age < 730.5


*** 7. MERGING HOUSEHOLD-LEVEL COVARIATES ***
//create rid variable to match children with households from HH file
gen rid = substr(tid, 1, 3) + "1" + substr(tid, 5, 5)

tempfile child036_allwaves_coded
save `child036_allwaves_coded', replace 

//load master household dataset 
cd "`PKH'"
use "data/coded/household_allwaves_master.dta", clear 

//keep only relevant household covariates 
keep rid* survey_round received* pkh* L07 hh_head_education hh_head_agr ///
	 hh_head_serv hh_educ* roof_type* wall_type* floor_type* clean_water own_latrine ///
	 square_latrine own_septic_tank electricity_PLN logpcexp* hhsize_ln *baseline_nm *miss ///
	 kabupaten kecamatan province split_indicator lack* low_* num_*_pc_bl total_*_pc_bl non_pkh_hh_cct_kec

tempfile hh_covariates
save `hh_covariates', replace

use `child036_allwaves_coded', clear 
//generate rid_merge variable to merge to correct household (accounts for HH splits after baseline)
gen rid_merge = substr(tid, 1, 3) + "1" + substr(tid,5,5) if survey_round == 0
replace rid_merge = substr(tid_w3, 1, 3) + "1" + substr(tid_w3,5,9) if survey_round == 1
replace rid_merge = substr(tid_hh_merge, 1, 3) + "1" + substr(tid_hh_merge,5,13) if survey_round == 2 //need to use tid_hh_merge variable for Wave IV
//merge 
merge m:1 rid_merge survey_round using `hh_covariates'
tab _merge
drop if _merge == 2
drop _merge 


//keep only PKH experimental sample 
drop if missing(L07)


*** 8. CREATE AGE BINS FOR CONTROLS ***
//month bins up to 1 year of age 
forvalues i = 0/11 {
	gen agebin_`i'month = 0 if !missing(age_months)
	replace agebin_`i'month = 1 if age_months == `i'
}

//quarter-year bins from year 1 to year 9
forvalues i = 1/9 {
	forvalues j = 1/4{
		gen agebin_`i'year_quarter`j' = 0 if !missing(age_months)
		replace agebin_`i'year_quarter`j' = 1 if (age_months - (12 * `i')) >= 3*(`j' - 1) & (age_months - (12 * `i')) < (3 * `j')
	}
}


*** 9. GENERATE KECAMATAN-LEVEL BASELINE AVERAGES OF OUTCOMES ***
local outcomes imm_age_complete imm_age_uptak_percent_only imm_age_uptak_percent_all ///
				imm_uptak_pct_23mons imm_uptak_pct_23mons_all ever_breastfed breastfed_1hr excl_breastfed_3mon ///
				mal_weightforage severe_weightforage ///
				mal_heightforage severe_heightforage mal_weightforheight severe_weightforheight ///
				stunted_neonatal severe_stunted_neonatal stunted_1to12months severe_stunted_1to12months ///
				stunted_1to2years severe_stunted_1to2years stunted_2to3years severe_stunted_2to3years ///
				stunted_3to4years severe_stunted_3to4years stunted_4to5years severe_stunted_4to5years ///
				diarrhea_lastmonth fevercough_lastmonth

//generate kecamatan-level averages of outcomes
foreach x of varlist `outcomes' {
	bysort kecamatan survey_round: egen `x'_kecw = mean(`x')
	bysort kecamatan (survey_round): gen `x'_kecbl = `x'_kecw[1] if survey_round[1] == 0
}


*** 10. MATCH INDIVIDUALS ACROSS WAVE AND GENERATE INDICATOR ***
//tempfile 
tempfile pre_cross_wave
save `pre_cross_wave', replace

//open Wave III survey
cd "`pkhdata'"
use "Wave III/Data/R_AR_01_fin.dta" 
duplicates drop
tostring rar00 rar00_07, replace format(%02.0f)

//generate Wave I tid string from roster variables
gen tid_w3 = substr(rid,1,3) + "4" + substr(rid,5,9) + rar00
gen tid_cross_wave = substr(tid_w3,1,9) + substr(tid_w3,14,2) if rar00_07 == "00"
replace tid_cross_wave = substr(tid_w3,1,9) + rar00_07 if rar00_07 != "00"

keep tid_w3 tid_cross_wave

tempfile wave3_roster
save `wave3_roster', replace

//merge with full dataset
use `pre_cross_wave', clear
merge m:1 tid_w3 using `wave3_roster'
drop if _merge == 2

//generate string to link individuals between Wave III and Wave IV
gen tid_w3_w4 = tid_w3 if survey_round == 1
gen indicator_temp = substr(tid_w4,12,2) if survey_round == 2
replace tid_w3_w4 = substr(tid_w4,1,7) + substr(tid_w4,10,6) + substr(tid_w4,18,2) if survey_round == 2 & indicator_temp != "13"
drop indicator_temp 

//use tid_w3_w4 variable as way to carry forward tid_cross_wave through survey rounds
replace tid_cross_wave = tid if survey_round == 0
replace tid_cross_wave = "" if survey_round == 1 & tcov3 == 3
bysort tid_w3_w4 (survey_round): carryforward tid_cross_wave if !missing(tid_w3_w4), replace

//generate baseline sample indicator
bysort tid_cross_wave (survey_round): gen in_baseline_sample = (survey_round[1] == 0)


//save
compress
cd "`PKH'data/coded"
save "child0to36months_allwaves_master", replace



