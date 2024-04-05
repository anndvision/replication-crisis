
*** 1. INTRODUCTION ***
/*
Description: 	Generates .tex file for Table 1 (First-Stage Regressions)
Uses: 			household_allwaves_master.dta
Creates: 		table1.tex
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
local latex = "`output'/latex"
local TARGET2 = "$TARGET2"

*** 3. PREPARE FOR REGRESSION ***
cd "`PKH'data/coded"
use "household_allwaves_master.dta", clear

//drop *_miss variables related to food/non-food expenditure
drop logpcexp_food_baseline_miss logpcexp_nonfood_baseline_miss

//create local with all baseline characteristics
local baseline_controls /// 
	  hh_head_agr_baseline_nm hh_head_serv_baseline_nm hh_educ*baseline_nm /// 
	  roof_type*baseline_nm wall_type*baseline_nm floor_type*baseline_nm clean_water_baseline_nm ///
	  own_latrine_baseline_nm square_latrine_baseline_nm own_septic_tank_baseline_nm /// 
	  electricity_PLN_baseline_nm logpcexp_baseline_nm hhsize_ln_baseline_nm
//removed all "*miss" variables in order to elimitate errant inclusion of "miss_i" variable, does not affect regressions since 
//ordinarily gets dropped for collinearity anyway


//generate interaction variables
foreach x of varlist `baseline_controls' {
	generate `x'_i = `x'*L07
}

//create local with all interactions 
local interactions *_i


//generate indicator variables for kabupaten fixed effects
tab kabupaten, gen(kabu_)

*** 4. FIRST STAGE REGRESSIONS ***
cd "`output'"

*INSTRUMENT: LOTTERY ONLY
//Wave III / midline 
eststo model1: reg pkh_by_this_wave L07 `baseline_controls' kabu_* ///
			   if survey_round == 1, vce(cluster kecamatan)

//add F-statistic
test L07
estadd scalar f_statistic `r(F)'

//add control mean
summ pkh_by_this_wave if L07 == 0 & survey_round == 1 & e(sample) == 1
estadd scalar control_mean `r(mean)'


//Wave IV / endline
eststo model2: reg pkh_by_this_wave L07 `baseline_controls' kabu_* ///
			   if survey_round == 2, vce(cluster kecamatan)

//add F-statistic
test L07
estadd scalar f_statistic `r(F)'

//add control mean
summ pkh_by_this_wave if L07 == 0 & survey_round == 2 & e(sample) == 1
estadd scalar control_mean `r(mean)'


*** 5. OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model1 model2
	using "table1.tex", booktabs label se
	keep(L07)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("2-Year" "6-Year", lhs("Outcome: Received CCT"))
	varlab(L07 "Treatment")
	r2
	nogaps
	scalars("control_mean Control Mean" "f_statistic F-statistic")
	nonotes
	fragment
	replace;

#delimit cr



