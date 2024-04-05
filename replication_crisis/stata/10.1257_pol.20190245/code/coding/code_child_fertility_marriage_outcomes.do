
*** 1. INTRODUCTION ***
/*

Description: 	Completes coding of childhood fertility and marriage outcomes for analysis
Uses: 			tracking_child6to15_attrition_for_fertility; raw survey data
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

//Set working directory:
cd "`pkhdata'"

*** 3. CODE MOTHERS WHO HAVE CHILDREN LIVING IN THE HOUSEHOLD (FOR FERTILITY) ***  
//open R_AR_01 (household roster) for Wave III 
use "Wave III/Data/R_AR_01_fin.dta", clear 
duplicates drop 

//convert rar00 variable to string 
tostring rar00, replace format(%02.0f)
//create id variable 
gen id = rid + rar00 
//create mother_id variable for merging
gen mother_id = id 
//create aid_original variable for merging to child attrition data 
gen aid_original = substr(id, 1, 3) + "3" + substr(id, 5, 11)

//keep observations who have mother living in the household
preserve
drop if (rar06 == 51 | rar06 == 52 | missing(rar06))
//convert rar06 (mother's serial number) to string
tostring rar06, replace format(%02.0f) force
drop mother_id 
gen mother_id = rid + rar06

//only need to keep one observation per mother id (could be multiple mothers within household) 
bysort mother_id (rar00): gen nth = _n
keep if nth == 1
drop nth

//only need to keep mother_id variable 
keep mother_id 
tempfile mother_ids
save `mother_ids', replace 

//now merge back into full roster
restore 
merge 1:1 mother_id using `mother_ids' // should not be any with _merge == 2

//generate "has child" variable but only for women 
gen has_child_in_hh = (_merge == 3) if rar03 == 3
drop _merge 


*MARRIAGE OUTCOMES
gen married = (rar07 == 2) if !missing(rar07)
gen married_div_widowed = (rar07 > 1) if !missing(rar07)

//generate survey round indicator 
gen survey_round = 1

//save tempfile
tempfile fertility_marriage_roster_w3
save `fertility_marriage_roster_w3', replace 


*Wave IV
//open R_AR_01 (household roster) for Wave IV 
use "Wave IV/R_AR_01.dta", clear 
duplicates drop 

//convert rar00 variable to string 
tostring rar00, replace format(%02.0f)
//create id variable for merging 
gen id = rid + rar00 
//create mother_id variable for merging within-roster
gen mother_id = id 
//create aid_original variable for merging to child attrition data 
gen aid_original = substr(id, 1, 3) + "3" + substr(id, 5, 15)

//keep observations who have mother living in the household
preserve
drop if (rar06 == 51 | rar06 == 52 | missing(rar06))
//convert rar06 (mother's serial number) to string
tostring rar06, replace format(%02.0f) force
drop mother_id 
gen mother_id = rid + rar06
//only need to keep one observation per mother id (could be multiple mothers within household) 
bysort mother_id (rar00): gen nth = _n
keep if nth == 1
drop nth
//only need to keep mother_id variable 
keep mother_id 

tempfile mother_ids
save `mother_ids', replace 

//now merge back into full roster
restore 
merge 1:1 mother_id using `mother_ids' // should not be any with _merge == 2

//generate "has child" variable but only for women 
gen has_child_in_hh = (_merge == 3) if rar03 == 3
drop _merge 

*MARRIAGE OUTCOMES
gen married = (rar07 == 2) if !missing(rar07)
gen married_div_widowed = (rar07 > 1) if !missing(rar07)

//generate survey round indicator 
gen survey_round = 2


//append Wave III roster 
append using `fertility_marriage_roster_w3', force 

//only want to keep necessary variables for merge 
keep aid_original mother_id survey_round has_child_in_hh* married married_div_widowed

//save full Wave III/Wave IV roster 
tempfile fertility_marriage_roster_all
save `fertility_marriage_roster_all', replace 


//now open child attrition file
cd "`PKH'data/coded"
use "tracking_child6to15_attrition_for_fertility.dta", clear  

//merge with fertility roster to identify which children who were 6-15 at baseline also have children
merge m:1 aid_original survey_round using `fertility_marriage_roster_all', force 
//if didn't merge from using then they weren't in the baseline child roster 
drop if _merge == 2
drop _merge 


*3 cases with child where baseline gender is coded male but correctly coded as female in wave IV
*These are data entry issues, it seems
gen gender_current_wave = air02
recode gender_current_wave (3 = 0) //0 for female, 1 for male
label values gender_current_wave gender 


//code age-specific variables
gen has_child_in_hh_16to17 = has_child_in_hh if age_years >= 16 & age_years <= 17
gen has_child_in_hh_18to21 = has_child_in_hh if age_years >= 18 & age_years <= 21
gen has_child_in_hh_18to22 = has_child_in_hh if age_years >= 18 & age_years <= 22

gen married_16to17 = married if age_years >= 16 & age_years <= 17
gen married_div_16to17 = married_div_widowed if age_years >= 16 & age_years <= 17

gen married_18to21 = married if age_years >= 18 & age_years <= 21
gen married_div_18to21 = married_div_widowed if age_years >= 18 & age_years <= 21

gen married_18to22 = married if age_years >= 18 & age_years <= 22
gen married_div_18to22 = married_div_widowed if age_years >= 18 & age_years <= 22


sort aid survey_round 
//save
compress
cd "`PKH'data/coded"
save "child6to15_fertility_marriage_outcomes", replace


