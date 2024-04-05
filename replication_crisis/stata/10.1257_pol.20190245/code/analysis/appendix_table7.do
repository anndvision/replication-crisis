
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
use "village_outcomes_master.dta", clear

//generate indicator variables for kabupaten fixed effects
tab kabupaten, gen(kabu_)

//generate single endogenous variable for regression 
gen endog = K09 if survey_round == 1
replace endog = K13 if survey_round == 2

//generate per-capita and per-household values of outcomes 
foreach x of varlist num_totaldoc_live num_totaldoc_prac num_totalmidwife_live num_totalmidwife_prac num_tradbirth_live num_tradbirth_prac total_primary total_secondary {
	//per household 
	gen phh_`x' = `x' / lid02 

	//log per household 
	gen lhh`x' = ln((`x' / lid02) + 1)

	//per capita 
	gen pp_`x' = `x' / lid01

	//log per capita 
	gen lpp`x' = ln((`x' / lid01) + 1)
}



*** 4. REGRESSIONS (PER-CAPITA) ***
cd "`output'"


**Outcome: Number of doctors living in village
//Midline 
eststo model_a: ivregress 2sls num_totaldoc_live_pc ///
	num_totaldoc_live_pc_bl_nm num_totaldoc_live_pc_bl_miss kabu_* (endog = L07) if survey_round == 1, vce(cluster kecamatan)

	//add control mean 
	summ num_totaldoc_live_pc if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Endline
eststo model_b: ivregress 2sls num_totaldoc_live_pc ///
	num_totaldoc_live_pc_bl_nm num_totaldoc_live_pc_bl_miss kabu_* (endog = L07) if survey_round == 2, vce(cluster kecamatan)

	//add control mean 
	summ num_totaldoc_live_pc if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'



**Outcomes: doctors practicing in village, midwives living in village, midwives practicing in village, birth attendants living in village,
* 			birth attendants practicing in village, # primary schools, # secondary schools
forv i = 1/7 {
	local depvar: word `i' of "num_totaldoc_prac" "num_totalmidwife_live" "num_totalmidwife_prac" "num_tradbirth_live" "num_tradbirth_prac" "total_primary" "total_secondary"

	//Midline, Lottery Only 
	eststo model`i'1: ivregress 2sls `depvar'_pc ///
		`depvar'_pc_bl_nm `depvar'_pc_bl_miss kabu_* (endog = L07) if survey_round == 1, vce(cluster kecamatan)

		//add control mean 
		summ `depvar'_pc if L07 == 0 & survey_round == 1 & e(sample) == 1
		estadd scalar control_mean `r(mean)'


	//Endline, Lottery Only 
	eststo model`i'2: ivregress 2sls `depvar'_pc ///
		`depvar'_pc_bl_nm `depvar'_pc_bl_miss kabu_* (endog = L07) if survey_round == 2, vce(cluster kecamatan)

		//add control mean 
		summ `depvar'_pc if L07 == 0 & survey_round == 2 & e(sample) == 1
		estadd scalar control_mean `r(mean)'

}



*** 5. OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model_b
	using "appendix_table7.tex", booktabs label se
	keep(endog)
	b(%12.5f)
	se(%12.5f)
	nostar
	mlabels("6-Year", lhs("Outcome:"))
	varlab(endog "\midrule Doctors living in village")
	stats(control_mean, labels(" ") fmt("%12.5f"))
	nogaps
	noobs
	nonotes
	nolines
	fragment
	replace;

#delimit cr 


forvalues i = 1/7 {
	local depvar: word `i' of "Doctors practicing in village" "Midwives living in village" "Midwives practicing in village" ///
			"Traditional birth attendants living in village" "Traditional birth attendants practicing in village" "Primary schools in village" "Secondary schools in village"
		
	#delimit ;

	esttab model`i'2
		using "appendix_table7.tex", booktabs label se
		keep(endog)
		b(%12.5f)
		se(%12.5f)
		nostar
		varlab(endog "\\ `depvar'")
		stats(control_mean, labels(" ") fmt("%12.5f"))
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




