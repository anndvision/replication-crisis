*** 1. INTRODUCTION ***
/*
Description: Calculates Romano-Wolf p-values for Table 5
Uses: 	child6to15_allwaves_master.dta
Creates: rwolf_pvalues_table5.xls
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


*** 2. TABLE 5 (IV EFFECT OF PKH ON CHILD LABOR) ***
//specify file for putexcel
putexcel set rwolf_pvalues_table5.xls, sheet(table5) replace

//set local with number of outcomes 
local num_outcomes = 2
//declare matrices for storing p-values
matrix pvals = J(`num_outcomes', 2, .)


*PANEL A 
cd "`PKH'data/coded"
use "child6to15_allwaves_master.dta", clear

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
	generate `x'_i = `x'*L07
}
local interactions *_i

//generate indicator variables for kabupaten fixed effects
tab kabupaten, gen(kabu_)


//subset for faster runtime, keeping only necessary variables and observations
drop if missing(L07) | missing(kecamatan)
keep wageonly_7to12 wageonly_20hrs_7to12 wageonly_13to15 wageonly_20hrs_13to15  ///
	`baseline_controls' `interactions' kabu* kecamatan survey_round pkh_by_this_wave L07


//local with outcome variables
local outcomes wageonly_13to15 wageonly_20hrs_13to15
matrix rownames pvals = `outcomes'

//change to output directory and run RW 
cd "`output'"

*** COLUMN 1 ***
//keep only survey_round == 1 for faster runtime 
preserve
keep if survey_round == 1

//Column 1 (2-year, Lottery Only)
rwolf_pkh `outcomes', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(20189762) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`num_outcomes' {
	local rowname: word `i' of `outcomes'
	matrix pvals[`i', 1] = e(rw_`rowname')
}



//Column 2 (6-year, Lottery Only)
restore
keep if survey_round == 2
rwolf_pkh `outcomes', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(78621005) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`num_outcomes' {
	local rowname: word `i' of `outcomes'
	matrix pvals[`i', 2] = e(rw_`rowname')
}


//export to excel 
putexcel A1 = matrix(pvals), names nformat(number_d3)

