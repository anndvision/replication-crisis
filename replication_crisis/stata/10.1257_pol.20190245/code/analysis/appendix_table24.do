

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

//generate necessary interaction dummies 
gen enroll_wagework_7to12 = enroll_age7to12 * wageonly_7to12
gen enroll_wagework_13to15 = enroll_age13to15 * wageonly_13to15

gen noenroll_wagework_7to12 = (1 - enroll_age7to12) * wageonly_7to12
gen noenroll_wagework_13to15 = (1 - enroll_age13to15) * wageonly_13to15


*** 4. PANEL A REGRESSIONS ***
cd "`output'"


*** 6. PANEL B REGRESSIONS (Ages 13-15) ***
**Outcome: Wage work last month
//Midline, Lottery Only 
eststo panelb_model_a: ivregress 2sls enroll_wagework_13to15 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)

	//add control mean 
	summ enroll_wagework_13to15 if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Endline, Lottery Only 
eststo panelb_model_b: ivregress 2sls enroll_wagework_13to15 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add control mean 
	summ enroll_wagework_13to15 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'



**Outcomes: Enrolled in SD (7-12), Attended school >85% last 2 weeks
forvalues i = 1/2 {
	local depvar: word `i' of "enroll_nowagework_13to15" "noenroll_wagework_13to15"
	
	forvalues j = 1/2 {

		eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == `j', vce(cluster kecamatan)

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == `j' & e(sample) == 1
		estadd scalar control_mean `r(mean)'
	}
}
	


*** 7. PANEL B OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab panelb_model_a panelb_model_b
	using "appendix_table24.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("2-Year" "6-Year", lhs("Outcome:"))
	varlab(pkh_by_this_wave "\midrule Enrolled in school $\times$ worked for wage last month")
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	noobs
	nonotes
	nolines
	fragment
	replace;

#delimit cr 


forvalues i = 1/2 {
	local depvar: word `i' of "Enrolled in school $\times$ no work for wage last month" "Not enrolled in school $\times$ worked for wage last month"

	#delimit ;

	esttab model`i'1 model`i'2
		using "appendix_table24.tex", booktabs label se
		keep(pkh_by_this_wave)
		b(%12.3f)
		se(%12.3f)
		nostar
		varlab(pkh_by_this_wave "\\ `depvar'")
		stats(control_mean, labels(" ") fmt("%12.3f"))
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

