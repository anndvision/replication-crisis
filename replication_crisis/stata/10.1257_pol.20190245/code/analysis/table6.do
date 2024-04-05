
*** 1. INTRODUCTION ***
/*
Description: 	Generates .tex file for Table 6 (Long-Run Outcomes)
Uses: 			child6to15_allwaves_master.dta, longrun_outcomes_master.dta
Creates: 		table6.tex
*/

*** 2. SET ENVIRONMENT ***
version 14.1 		
set more off		
clear all
set matsize 10000			
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

*** 3. IMPORT ADJUSTED P-VALUES FROM R-W OUTPUT ***
cd "`PKH'output"
//import excel document 
import excel using "rwolf_pvalues_table6.xls", firstrow clear
//loop over number of variables 
qui count 
local num_vars = `r(N)'
//loop for each variable
forv i = 1/`num_vars' {
	local var_name = A[`i']
	//loop for each column
	forv j = 1/2 {
		local `var_name'_p_`j': di %05.3f c`j'[`i'] // round to 3 decimal places
		//change 0 to <0.001
		if ``var_name'_p_`j'' == 0 | "``var_name'_p_`j''" == "0.000" {
			local `var_name'_p_`j' = "[" + "<0.001" + "]"
		}
		else {
			local `var_name'_p_`j' = "[" + "``var_name'_p_`j''" + "]"
		}
	}
}

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

//now also generate interactions with survey round variable for "stack and cluster" approach 
//to calculating 2-vs-6-year p-values
//generate dummy for survey round 
gen survey_round_1 = (survey_round == 1) if survey_round != 0
gen survey_round_2 = (survey_round == 2) if survey_round != 0

//baseline controls, endogenous regressor, and instrument 
foreach var of varlist `baseline_controls' {
	gen `var'1 = `var'*survey_round_1
	gen `var'2 = `var'*survey_round_2

	local baseline_controls_1 `baseline_controls_1' `var'1
	local baseline_controls_2 `baseline_controls_2' `var'2
}

//to be able to call in program for p-values 
global baseline_controls_1 `baseline_controls_1'
global baseline_controls_2 `baseline_controls_2'

//fixed effects
foreach var of varlist kabu_* {
	gen d`var'_1 = `var'*survey_round_1
	gen d`var'_2 = `var'*survey_round_2
}

//endogenous regressors (don't include in loop above so that they don't get added to baseline controls local)
foreach var of varlist pkh_by_this_wave L07 {
	gen `var'1 = `var'*survey_round_1
	gen `var'2 = `var'*survey_round_2
}


//generate dummy version of PKH variable to be able to omit certain output 
gen pkh_by_this_wave_new = pkh_by_this_wave

*** 4. PANEL A REGRESSIONS ***
cd "`output'"

**Outcome: Enrollment in any school
//Midline, Lottery Only 
eststo model_a: ivregress 2sls enroll_any_15to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)

	//add RW adjusted p-value
	estadd local rwpval "`enroll_any_15to17_p_1'"

	//add control mean 
	summ enroll_any_15to17 if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Endline, Lottery Only 
eststo model_b: ivregress 2sls enroll_any_15to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add RW adjusted p-value
	estadd local rwpval "`enroll_any_15to17_p_2'"

	//add control mean 
	summ enroll_any_15to17 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Calculate 2-vs-6-year p-value! 
eststo pval_26: ivregress 2sls enroll_any_15to17 `baseline_controls_1' `baseline_controls_2' /// 
	dkabu_* (pkh_by_this_wave? = L07?), vce(cluster kecamatan) 

	//test 2 vs. 6 year 
	test pkh_by_this_wave1=pkh_by_this_wave2
	estadd scalar pval26 `r(p)'



**Outcome: completed high school 18-21,....
forvalues i = 1/2 {	
	local depvar: word `i' of "enroll_SMA_15to17" "completed_SMA_18to21" ///

	forvalues j = 1/2 {
		if `i' == 2 & `j' == 1 {
			local pkhdef = "pkh_by_this_wave_new"
		}
		else {
			local pkhdef = "pkh_by_this_wave"
		}

		eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(`pkhdef' = L07) if survey_round == `j', vce(cluster kecamatan)

		if `i' != 2 | `j' > 1 {
			//add RW adjusted p-value
			estadd local rwpval "``depvar'_p_`j''"

			//add control mean 
			summ `depvar' if L07 == 0 & survey_round == `j' & e(sample) == 1
			estadd scalar control_mean `r(mean)'
		}
	}

	//calculate 2-vs-6-year p-value for enrollment outcome only
	if `i' == 1 {
		//Calculate 2-vs-6-year p-value! 
		eststo pval_26_`i': ivregress 2sls `depvar' `baseline_controls_1' `baseline_controls_2' /// 
		dkabu_* (pkh_by_this_wave? = L07?), vce(cluster kecamatan)

		//test 2 vs 6 year 
		test pkh_by_this_wave1 = pkh_by_this_wave2
		estadd scalar pval26 `r(p)'	
	}
	else {
		eststo pval_26_`i': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(pkh_by_this_wave_new = L07) if survey_round == 1, vce(cluster kecamatan)
	}

}


*** 5. PANEL A OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model_a model_b pval_26
	using "table6.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("2-Year" "6-Year" "p-value (2-Yr. = 6-Yr.)", lhs("Outcome:"))
	varlab(pkh_by_this_wave "Enrolled in school (Ages 15-17)")
	refcat(pkh_by_this_wave "\midrule \emph{Panel A: School Enrollment/Completion Outcomes}", nolabel)
	scalars("rwpval \ " "control_mean \ " "pval26 \ ")
	nogaps
	noobs
	nonotes
	nolines
	fragment
	replace;

#delimit cr 

forvalues i = 1/2 {
	local depvar: word `i' of "Enrolled in high school (Ages 15-17)" "Completed high school (Ages 18-21)" ///

	#delimit ;

	esttab model`i'1 model`i'2  pval_26_`i'
		using "table6.tex", booktabs label se
		keep(pkh_by_this_wave)
		b(%12.3f)
		se(%12.3f)
		nostar
		varlab(pkh_by_this_wave "\\ `depvar'")
		scalars("rwpval \ " "control_mean \ " "pval26 \ ")
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

*** 6. PANEL B REGRESSIONS ***
cd "`PKH'output"

**Outcome: wageonly_16to17
//Midline, Lottery Only 
eststo panelb_model_a: ivregress 2sls wageonly_16to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)

	//add RW adjusted p-value
	estadd local rwpval "`wageonly_16to17_p_1'"

	//add control mean 
	summ wageonly_16to17 if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Endline, Lottery Only 
eststo panelb_model_b: ivregress 2sls wageonly_16to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add RW adjusted p-value
	estadd local rwpval "`wageonly_16to17_p_2'"

	//add control mean 
	summ wageonly_16to17 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Calculate 2-vs-6-year p-value! 
eststo pval_26: ivregress 2sls wageonly_16to17 `baseline_controls_1' `baseline_controls_2' /// 
	dkabu_* (pkh_by_this_wave? = L07?), vce(cluster kecamatan) 

	//test 2 vs. 6 year 
	test pkh_by_this_wave1=pkh_by_this_wave2
	estadd scalar pval26 `r(p)'


**Outcome: 20+ hours (16-17), any wage work 18-21, wage work 20+ hours 18-21
forvalues i = 1/3 {
	local depvar: word `i' of "wageonly_20hrs_16to17" "wageonly_18to21" "wageonly_20hrs_18to21"
	
	forvalues j = 1/2 {
		if (`i' == 2 | `i' == 3) & `j' == 1 {
			local pkhdef = "pkh_by_this_wave_new"
		}
		else {
			local pkhdef = "pkh_by_this_wave"
		}

		eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(`pkhdef' = L07) if survey_round == `j', vce(cluster kecamatan)


		if `i' == 1 | `j' > 1 {
			//add RW adjusted p-value
			estadd local rwpval "``depvar'_p_`j''"

			//add control mean 
			summ `depvar' if L07 == 0 & survey_round == `j' & e(sample) == 1
			estadd scalar control_mean `r(mean)'
		}
	}

	//calculate 2-vs-6-year p-value for enrollment outcome only
	if `i' == 1 {
		//Calculate 2-vs-6-year p-value! 
		eststo pval_26_`i': ivregress 2sls `depvar' `baseline_controls_1' `baseline_controls_2' /// 
		dkabu_* (pkh_by_this_wave? = L07?), vce(cluster kecamatan)

		//test 2 vs 6 year 
		test pkh_by_this_wave1 = pkh_by_this_wave2
		estadd scalar pval26 `r(p)'	
	}
	else {
		eststo pval_26_`i': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(pkh_by_this_wave_new = L07) if survey_round == 1, vce(cluster kecamatan)
	}

}


*** 7. PANEL B OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab panelb_model_a panelb_model_b pval_26
	using "table6.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	nomtitles
	varlab(pkh_by_this_wave "Worked for wage last month (Ages 16-17)")
	refcat(pkh_by_this_wave "\\ \emph{Panel B: Labor Outcomes (Ages 16-21)}", nolabel)
	scalars("rwpval \ " "control_mean \ " "pval26 \ ")
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;

#delimit cr 

forvalues i = 1/3 {
	local depvar: word `i' of "Worked 20+ hours for wage last month (Ages 16-17)" ///
								"Worked for wage last month (Ages 18-21)" "Worked 20+ hours for wage last month (Ages 18-21)"
	
	#delimit ;

	esttab model`i'1 model`i'2 pval_26_`i'
		using "table6.tex", booktabs label se
		keep(pkh_by_this_wave)
		b(%12.3f)
		se(%12.3f)
		nostar
		varlab(pkh_by_this_wave "\\ `depvar'")
		scalars("rwpval \ " "control_mean \ " "pval26 \ ")
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


*** 8. PANEL C REGRESSIONS ***
cd "`PKH'data/coded"
use "child6to15_fertility_marriage_outcomes.dta", clear

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

//now also generate interactions with survey round variable for "stack and cluster" approach 
//to calculating 2-vs-6-year p-values
//generate dummy for survey round 
gen survey_round_1 = (survey_round == 1) if survey_round != 0
gen survey_round_2 = (survey_round == 2) if survey_round != 0

//baseline controls, endogenous regressor, and instrument 
foreach var of varlist `baseline_controls' {
	gen `var'1 = `var'*survey_round_1
	gen `var'2 = `var'*survey_round_2

	local baseline_controls_1 `baseline_controls_1' `var'1
	local baseline_controls_2 `baseline_controls_2' `var'2
}

//to be able to call in program for p-values 
global baseline_controls_1 `baseline_controls_1'
global baseline_controls_2 `baseline_controls_2'

//fixed effects
foreach var of varlist kabu_* {
	gen d`var'_1 = `var'*survey_round_1
	gen d`var'_2 = `var'*survey_round_2
}

//endogenous regressors (don't include in loop above so that they don't get added to baseline controls local)
foreach var of varlist pkh_by_this_wave L07 {
	gen `var'1 = `var'*survey_round_1
	gen `var'2 = `var'*survey_round_2
}

//generate dummy version of PKH variable to be able to omit certain output 
gen pkh_by_this_wave_new = pkh_by_this_wave

cd "`output'"

**Outcome: Married (Age 16-17)
//Midline, Lottery Only 
eststo panelc_model_a: ivregress 2sls married_16to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)

	//add RW adjusted p-value
	estadd local rwpval "`married_16to17_p_1'"

	//add control mean 
	summ married_16to17 if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'

//Endline, Lottery Only 
eststo panelc_model_b: ivregress 2sls married_16to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add RW adjusted p-value
	estadd local rwpval "`married_16to17_p_2'"

	//add control mean 
	summ married_16to17 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'

//Calculate 2-vs-6-year p-value! 
eststo pval_26: ivregress 2sls married_16to17 `baseline_controls_1' `baseline_controls_2' /// 
	dkabu_* (pkh_by_this_wave? = L07?), vce(cluster kecamatan) 

	//test 2 vs. 6 year 
	test pkh_by_this_wave1=pkh_by_this_wave2
	estadd scalar pval26 `r(p)'


**Outcome: married_18to21
	local depvar = "married_18to21"
	
	forvalues j = 1/2 {
		if `j' == 1 {
			local pkhdef = "pkh_by_this_wave_new"
		}
		else {
			local pkhdef = "pkh_by_this_wave"
		}

	eststo modelc1`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
	(`pkhdef' = L07) if survey_round == `j', vce(cluster kecamatan)

	if `j' > 1 {
		//add RW adjusted p-value
		estadd local rwpval "``depvar'_p_`j''"

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == `j' & e(sample) == 1
		estadd scalar control_mean `r(mean)'
	}

	eststo pval_26_`i': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
	(pkh_by_this_wave_new = L07) if survey_round == 1, vce(cluster kecamatan)

}


*** 9. PANEL C OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab panelc_model_a panelc_model_b pval_26
	using "table6.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	nomtitles
	varlab(pkh_by_this_wave "Married (Ages 16-17)")
	refcat(pkh_by_this_wave "\\ \emph{Panel C: Marriage Outcomes (Ages 16-21)}", nolabel)
	scalars("rwpval \ " "control_mean \ " "pval26 \ ")
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;

#delimit cr 

local depvar = "Married (Ages 18-21)"
	
#delimit ;

esttab modelc11 modelc12 pval_26_`i'
	using "table6.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	varlab(pkh_by_this_wave "\\ `depvar'")
	scalars("rwpval \ " "control_mean \ " "pval26 \ ")
	nomtitles
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;


#delimit cr


