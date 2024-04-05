
*** 1. INTRODUCTION ***
/*
Description: 	Generates .tex file for Appendix Table 9
Uses: 			child6to15_allwaves_master.dta
Creates: 		appendix_table9.tex
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

*** 4. PANEL A REGRESSIONS ***
cd "`output'"

**Outcome: Enrollment in any school
//Midline, Lottery Only 
eststo model_a: ivregress 2sls enroll_7to15 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)

	//add control mean 
	summ enroll_7to15 if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'

//Midline, Lottery x Assets 
eststo model_b: ivregress 2sls enroll_7to15 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07 `interactions') if survey_round == 1, vce(cluster kecamatan)

	//add control mean 
	summ enroll_7to15 if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'

//Endline, Lottery Only 
eststo model_c: ivregress 2sls enroll_7to15 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add control mean 
	summ enroll_7to15 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'

//Endline, Lottery x Assets
eststo model_d: ivregress 2sls enroll_7to15 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07 `interactions') if survey_round == 2, vce(cluster kecamatan)

	//add control mean 
	summ enroll_7to15 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


**Outcomes: Overall >85% attend, overall percentage attendance
forvalues i = 1/1 {
	local depvar: word `i' of "age7to15_85_twoweeks"
	
	forvalues j = 1/4 {
		if `j' == 1 | `j' == 3 {
			local instrument = "L07"
		}
		else {
			local instrument = "L07 `interactions'"
		}
		if `j' <= 2 {
			local round = "1"
		}
		else {
			local round = "2"
		}

	eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = `instrument') if survey_round == `round', vce(cluster kecamatan)

	//add control mean 
	summ `depvar' if L07 == 0 & survey_round == `round' & e(sample) == 1
	estadd scalar control_mean `r(mean)'
	}

}


*** 5. PANEL A OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model_a model_b model_c model_d
	using "appendix_table9.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("Lottery Only" "Lottery + Assets" "Lottery Only" "Lottery + Assets", lhs("Outcome:"))
	mgroups("2-Year" "6-Year", pattern(1 0 1 0) span prefix(\multicolumn{@span}{c}{) suffix(}) erepeat(\cmidrule(lr){@span}))
	varlab(pkh_by_this_wave "\midrule Enrolled in school (any level)")
	scalars("control_mean \ ")
	nogaps
	noobs
	nonotes
	nolines
	fragment
	replace;

#delimit cr 

forvalues i = 1/1 {
	local depvar: word `i' of ">85\% attendance last two weeks"
	
	#delimit ;

	esttab model`i'1 model`i'2 model`i'3 model`i'4
		using "appendix_table9.tex", booktabs label se
		keep(pkh_by_this_wave)
		b(%12.3f)
		se(%12.3f)
		nostar
		varlab(pkh_by_this_wave "\\ `depvar'")
		scalars("control_mean \ ")
		nomtitles
		nogaps
		nonumbers
		noobs
		nonotes
		nolines
		fragment
		append;

		#delimit cr
}

#delimit cr


