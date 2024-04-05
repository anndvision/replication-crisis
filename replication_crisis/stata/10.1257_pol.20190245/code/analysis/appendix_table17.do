
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
	generate `x'_i = `x'*L07
}
local interactions *_i

//generate indicator variables for kabupaten fixed effects
tab kabupaten, gen(kabu_)

*** 4. PANEL A REGRESSIONS ***
cd "`output'"


//one loop for each gender 
forvalues x = 1/2 {

	local panellabel: word `x' of "\midrule \emph{Panel A: Boys 0-60 months}" "\\ \emph{Panel B: Girls 0-60 months}"
	//for esttab command: only "replace" on first loop through
	local replaceappend: word `x' of "replace" "append"

	**Outcome: Stunting
	//Midline, Lottery Only 
	eststo model_`x'a: ivregress 2sls mal_heightforage `baseline_controls' agebin* kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == 1 & gender == `x', vce(cluster kecamatan)

		//add control mean 
		summ mal_heightforage if L07 == 0 & survey_round == 1 & gender == `x' & e(sample) == 1
		estadd scalar control_mean `r(mean)'


	//Endline, Lottery Only 
	eststo model_`x'b: ivregress 2sls mal_heightforage `baseline_controls' agebin* kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == 2 & gender == `x', vce(cluster kecamatan)

		//add control mean 
		summ mal_heightforage if L07 == 0 & survey_round == 2 & gender == `x' & e(sample) == 1
		estadd scalar control_mean `r(mean)'


	**Outcomes: severe_heightforage, mal_weightforheight, severe_weightforheight
	forvalues i = 1/3 {
		local depvar: word `i' of "severe_heightforage" "mal_weightforage" "severe_weightforage"
		
		forvalues j = 1/2 {

			eststo model`x'`i'`j': ivregress 2sls `depvar' `baseline_controls' agebin* kabu_* /// 
			(pkh_by_this_wave = L07) if survey_round == `j' & gender == `x', vce(cluster kecamatan)


			//add control mean 
			summ `depvar' if L07 == 0 & survey_round == `j' & gender == `x' & e(sample) == 1
			estadd scalar control_mean `r(mean)'
		}

	}


	*** 5. PANEL A OUTPUT TO LATEX ***
	cd "`latex'"

	if `x' == 1 {
		#delimit ;
		esttab model_`x'a model_`x'b
			using "appendix_table17.tex", booktabs label se
			keep(pkh_by_this_wave)
			b(%12.3f)
			se(%12.3f)
			nostar
			mlabels("2-Year" "6-Year", lhs("Outcome:"))
			varlab(pkh_by_this_wave "Stunted")
			refcat(pkh_by_this_wave "`panellabel'", nolabel)
			stats(control_mean, labels(" ") fmt("%12.3f"))
			nogaps
			noobs
			nonotes
			nolines
			fragment
			replace;

		#delimit cr 
	}
	else if `x' == 2 {
		#delimit ;
		esttab model_`x'a model_`x'b
			using "appendix_table17.tex", booktabs label se
			keep(pkh_by_this_wave)
			b(%12.3f)
			se(%12.3f)
			nostar
			nomtitles
			varlab(pkh_by_this_wave "Stunted")
			refcat(pkh_by_this_wave "`panellabel'", nolabel)
			stats(control_mean, labels(" ") fmt("%12.3f"))
			nogaps
			nonumbers
			noobs
			nonotes
			nolines
			fragment
			append;

		#delimit cr 
	} 

	forvalues i = 1/3 {
		local depvar: word `i' of "Severely stunted" "Malnourished" ///
									"Severely malnourished"
		
		#delimit ;

		esttab model`x'`i'1 model`x'`i'2
			using "appendix_table17.tex", booktabs label se
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



}

#delimit cr



