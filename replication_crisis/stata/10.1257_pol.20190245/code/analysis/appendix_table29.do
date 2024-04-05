
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

//drop above 99th percentile values for pcexp by survey_round
drop if survey_round == 1 & pcexp >= 748321.4
drop if survey_round == 2 & pcexp >= 1323839

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

**Outcome: Log per-capita expenditure (Jakarta + West Java)
//Midline, Lottery Only 
eststo model_a: ivregress 2sls logpcexp `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1 & (province == 31 | province == 32), vce(cluster kecamatan)

	//add control mean 
	summ logpcexp if L07 == 0 & survey_round == 1 & (province == 31 | province == 32) & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Endline, Lottery Only 
eststo model_b: ivregress 2sls logpcexp `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & (province == 31 | province == 32), vce(cluster kecamatan)

	//add control mean 
	summ logpcexp if L07 == 0 & survey_round == 2 & (province == 31 | province == 32) & e(sample) == 1
	estadd scalar control_mean `r(mean)'


**Outcomes: East Java, NTT, North Sulawesi & Gorontalo
forvalues i = 1/3 {
	local depvar = "logpcexp"
	local ifcondition: word `i' of "province == 35" "province == 53" "(province == 71 | province == 75)"
	
	forvalues j = 1/2 {

		eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == `j' & `ifcondition', vce(cluster kecamatan)

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == `j' & `ifcondition' & e(sample) == 1
		estadd scalar control_mean `r(mean)'
	}

}



*** 5. PANEL A OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model_a model_b
	using "appendix_table29.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("2-Year" "6-Year", lhs("Outcome:"))
	varlab(pkh_by_this_wave "\midrule Log per-capita expenditure (DKI Jakarta \& West Java)")
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	noobs
	nonotes
	nolines
	fragment
	replace;

#delimit cr 

forvalues i = 1/3 {
	local depvar: word `i' of "Log per-capita expenditure (East Java)" "Log per-capita expenditure (East Nusa Tenggara)" ///
								"Log per-capita expenditure (North Sulawesi \& Gorontalo)"
	
	#delimit ;

	esttab model`i'1 model`i'2
		using "appendix_table29.tex", booktabs label se
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


