
*** 1. INTRODUCTION ***
/*
Description: Calculates Romano-Wolf p-values for Table 2
Uses: marriedwoman16to49_allwaves_master; child0to36months_allwaves_master	
Creates: rwolf_pvalues_table2.xls
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


//set ado path for rwolf_pkh
adopath + "`PKH'ado"


*** 2. TABLE 2 (HEALTH-SEEKING BEHAVIORS) ***
//specify file for putexcel
putexcel set rwolf_pvalues_table2.xls, sheet(table2) replace

//set local with number of outcomes 
local num_outcomes = 8
//declare matrices for storing p-values
matrix pvals = J(`num_outcomes', 2, .)


**PANEL A DATA** 
cd "`PKH'data/coded"
use "marriedwoman16to49_allwaves_master.dta", clear

//drop *_miss variables related to food/non-food expenditure
drop logpcexp_food_baseline_miss logpcexp_nonfood_baseline_miss

//create local with all baseline characteristics (INCLUDES LOG PCE)
local baseline_controls /// 
	  hh_head_agr_baseline_nm hh_head_serv_baseline_nm hh_educ*baseline_nm /// 
	  roof_type*baseline_nm wall_type*baseline_nm floor_type*baseline_nm clean_water_baseline_nm ///
	  own_latrine_baseline_nm square_latrine_baseline_nm own_septic_tank_baseline_nm /// 
	  electricity_PLN_baseline_nm hhsize_ln_baseline_nm logpcexp_baseline_nm *miss

//generate interaction variables
drop *_i //drop 2 existing panel variables ending in *_i
foreach x of varlist `baseline_controls' {
	qui generate `x'_i = `x'*L07
}
local interactions *_i

//generate indicator variables for kabupaten fixed effects
qui tab kabupaten, gen(kabu_)

//tempfile to append later 
tempfile panel_a
save `panel_a', replace

*PANEL B OUTCOMES - APPEND TO PANEL A
use "child0to36months_allwaves_master.dta", clear

//drop *_miss variables related to food/non-food expenditure
drop logpcexp_food_baseline_miss logpcexp_nonfood_baseline_miss

//create local with all baseline characteristics (INCLUDES LOG PCE)
local baseline_controls /// 
	  hh_head_agr_baseline_nm hh_head_serv_baseline_nm hh_educ*baseline_nm /// 
	  roof_type*baseline_nm wall_type*baseline_nm floor_type*baseline_nm clean_water_baseline_nm ///
	  own_latrine_baseline_nm square_latrine_baseline_nm own_septic_tank_baseline_nm /// 
	  electricity_PLN_baseline_nm hhsize_ln_baseline_nm logpcexp_baseline_nm *miss

//generate interaction variables
foreach x of varlist `baseline_controls' {
	qui generate `x'_i = `x'*L07
}
local interactions *_i

//generate indicator variables for kabupaten fixed effects
qui tab kabupaten, gen(kabu_)

//append panel A variables to dataset 
append using `panel_a'
local interactions *_i

//subset for faster runtime, keeping only necessary variables and observations
drop if missing(L07) | missing(kecamatan)
keep pre_natal_visits good_assisted_delivery delivery_facility post_natal_visits iron_pills_dummy ///
	 imm_age_uptak_percent_only vitA_total_6mons_2years times_weighed_0to5 ///
	`baseline_controls' `interactions' kabu* agebin* kecamatan survey_round pkh_by_this_wave L07


//local with outcome variables
local outcomes pre_natal_visits good_assisted_delivery delivery_facility post_natal_visits iron_pills_dummy imm_age_uptak_percent_only vitA_total_6mons_2years times_weighed_0to5
matrix rownames pvals = `outcomes'




//change to output directory and run RW 
cd "`output'"

**Column 1 (2-year, Lottery Only)
//keep only survey_round == 1 for faster runtime 
preserve
keep if survey_round == 1

rwolf_pkh `outcomes', indepvar(pkh_by_this_wave) agebinvars(imm_age_uptak_percent_only vitA_total_6mons_2years times_weighed_0to5) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(21893202) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`num_outcomes' {
	local rowname: word `i' of `outcomes'
	matrix pvals[`i', 1] = e(rw_`rowname')
}

**Column 2 (6-year, Lottery Only)
restore
keep if survey_round == 2
rwolf_pkh `outcomes', indepvar(pkh_by_this_wave) agebinvars(imm_age_uptak_percent_only vitA_total_6mons_2years times_weighed_0to5) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(89210015) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`num_outcomes' {
	local rowname: word `i' of `outcomes'
	matrix pvals[`i', 2] = e(rw_`rowname')
}


//export to excel 
putexcel A1 = matrix(pvals), names nformat(number_d3)

