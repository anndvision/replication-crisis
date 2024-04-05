
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

**Outcome: Worked for home business last month (16-17)
//Full Sample, Lottery Only 
eststo model_a: ivregress 2sls homebusiness_16to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add control mean 
	summ homebusiness_16to17 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Boys, Lottery Only 
eststo model_b: ivregress 2sls homebusiness_16to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & air02 == 1, vce(cluster kecamatan)

	//add control mean 
	summ homebusiness_16to17 if L07 == 0 & survey_round == 2 & e(sample) == 1 & air02 == 1
	estadd scalar control_mean `r(mean)'


//Girls, Lottery Only 
eststo model_c: ivregress 2sls homebusiness_16to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & air02 == 3, vce(cluster kecamatan)

	//add control mean 
	summ homebusiness_16to17 if L07 == 0 & survey_round == 2 & e(sample) == 1 & air02 == 3
	estadd scalar control_mean `r(mean)'


**Outcome: home business 20+ last month, helped at home last month, helped 20+ hours at home last month
forvalues i = 1/3 {

	local depvar: word `i' of "homebusiness_20hrs_16to17" "helphome_16to17" "helphome_20hrs_16to17"
	
	forvalues j = 1/3 {
		if `j' == 1 {
			local gendercond = ""
		}
		else if `j' == 2 {
			local gendercond = "& air02 == 1"
		}
		else {
			local gendercond = "& air02 == 3"
		}

		eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == 2 `gendercond', vce(cluster kecamatan)

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == 2 & e(sample) == 1 `gendercond'
		estadd scalar control_mean `r(mean)'
	}

}



*** 5. PANEL A OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model_a model_b model_c
	using "appendix_table27.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("Full Sample" "Boys" "Girls", lhs("Outcome:"))
	varlab(pkh_by_this_wave "Worked for family business last month")
	refcat(pkh_by_this_wave "\midrule \emph{Panel A: Outcomes for Ages 16-17}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	noobs
	nonotes
	nolines
	fragment
	replace;

#delimit cr 

forvalues i = 1/3 {
	local depvar: word `i' of "Worked 20+ hrs. for family last month" "Helped at home last month" "Helped 20+ hours at home last month"

	#delimit ;

	esttab model`i'1 model`i'2 model`i'3
		using "appendix_table27.tex", booktabs label se
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

*** 6. PANEL B REGRESSIONS (Ages 18-21) ***
**Outcome: Family business work last month
//Full Sample, Lottery Only 
eststo panelb_model_a: ivregress 2sls homebusiness_18to21 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add control mean 
	summ homebusiness_18to21 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'

//Boys, Lottery Only 
eststo panelb_model_b: ivregress 2sls homebusiness_18to21 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & air02 == 1, vce(cluster kecamatan)

	//add control mean 
	summ homebusiness_18to21 if L07 == 0 & survey_round == 2 & e(sample) == 1 & air02 == 1
	estadd scalar control_mean `r(mean)'

//Girls, Lottery Only 
eststo panelb_model_c: ivregress 2sls homebusiness_18to21 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & air02 == 3, vce(cluster kecamatan)

	//add control mean 
	summ homebusiness_18to21 if L07 == 0 & survey_round == 2 & e(sample) == 1 & air02 == 3
	estadd scalar control_mean `r(mean)'


**Outcomes: family busines 20+ hours last month, helped at home last month, helped 20+ hours last month
forvalues i = 1/3 {

	local depvar: word `i' of "homebusiness_20hrs_18to21" "helphome_18to21" "helphome_20hrs_18to21"
	
	forvalues j = 1/3 {
		if `j' == 1 {
			local gendercond = ""
		}
		else if `j' == 2 {
			local gendercond = "& air02 == 1"
		}
		else {
			local gendercond = "& air02 == 3"
		}

		eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == 2 `gendercond', vce(cluster kecamatan)

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == 2 & e(sample) == 1 `gendercond'
		estadd scalar control_mean `r(mean)'
	}
}


*** 7. PANEL B OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab panelb_model_a panelb_model_b panelb_model_c
	using "appendix_table27.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	nomtitles
	varlab(pkh_by_this_wave "Worked for family business last month")
	refcat(pkh_by_this_wave "\\ \emph{Panel B: Outcomes for Ages 18-21}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;

#delimit cr 

forvalues i = 1/3 {
	local depvar: word `i' of "Worked 20+ hrs. for family last month" "Helped at home last month" "Helped 20+ hours at home last month"

	#delimit ;

	esttab model`i'1 model`i'2 model`i'3
		using "appendix_table27.tex", booktabs label se
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


