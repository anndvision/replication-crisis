
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
use "tracking_child6to15_attrition.dta", clear

//drop *_miss variables related to food/non-food expenditure
drop logpcexp_food_baseline_miss logpcexp_nonfood_baseline_miss

//create local with all baseline characteristics
local baseline_controls /// 
	  hh_head_agr_baseline_nm hh_head_serv_baseline_nm hh_educ*baseline_nm /// 
	  roof_type*baseline_nm wall_type*baseline_nm floor_type*baseline_nm clean_water_baseline_nm ///
	  own_latrine_baseline_nm square_latrine_baseline_nm own_septic_tank_baseline_nm /// 
	  electricity_PLN_baseline_nm logpcexp_baseline_nm hhsize_ln_baseline_nm *miss

//generate interaction variables
foreach x of varlist `baseline_controls' {
	generate `x'_i = `x'*L07
}

//create local with all interactions 
local interactions *_i

//generate indicator variables for kabupaten fixed effects
tab kabupaten, gen(kabu_)






**Table 5a: raw counts
estpost ttest surveyed if survey_round == 2, by(L07)
matrix total_count = e(count)
matrix control_count = e(count)
matrix treat_count = e(count)
matrix p_value = e(count)
matrix p_value_fe = e(count)

reg surveyed L07 kabu_* if survey_round == 2, vce(cluster kecamatan)
local p_fe = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))
matrix p_value_fe[1,1] = `p_fe'

eststo row1: reg surveyed L07 if survey_round == 2, vce(cluster kecamatan)
local p = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))

matrix p_value[1,1] = `p'
estadd matrix p_value
estadd matrix p_value_fe

summ surveyed if survey_round == 2
matrix total_count[1,1] = round(r(mean), .001)
estadd matrix total_count 

summ surveyed if survey_round == 2 & L07 == 0
matrix control_count[1,1] = round(r(mean), .001)
estadd matrix control_count 

summ surveyed if survey_round == 2 & L07 == 1
matrix treat_count[1,1] = round(r(mean), .001)
estadd matrix treat_count

//output to latex 
cd "`latex'"

#delimit ; 

esttab row1 using "appendix_table5a.tex", booktabs
	label
	cells("total_count control_count treat_count p_value(fmt(3)) p_value_fe(fmt(3))") 
	unstack 
	nonotes 
	nonumber 
	nomtitle 
	collabels("Overall" "Control" "Treatment" "\shortstack{Control - Treatment \\ \emph{p}-value}" "\shortstack{Control - Treatment \\ \emph{p}-value (District FE)}", lhs("Outcome:")) 
	noobs 
	varlabels(surveyed "Surveyed")
	fragment
	replace;

#delimit cr 

//loop over other variables of interest 
forvalues i = 1/10 {

	local tabvar: word `i' of "migrated_midline" "migrated_endline" "moved_away_midline" "moved_away_endline" ///
							 "died_midline" "died_endline" "panel_in_roster_midline" "panel_in_roster_endline" ///
							 	"household_attrited" "unaccounted_for"

	local tablabel: word `i' of "Migrated in Last 12 Months, 2-Year" "Migrated in Last 12 Months, 6-Year" ///
								"Migrated Prior to Last 12 Months, 2-Year" "Migrated Prior to Last 12 Months, 6-Year" ///
								"Died 2-Year" "Died 6-Year" "Still in Household Roster, 2-Year" ///
								"Still in Household Roster, 6-Year" "Household Attrited" "Unaccounted For"

	local row: word `i' of "A" "B" "C" "D" "E" "F" "G" "H" "I" "J"

	estpost ttest `tabvar' if survey_round == 2, by(L07)
	matrix total_count = e(count)
	matrix control_count = e(count)
	matrix treat_count = e(count)
	matrix p_value = e(count)
	matrix p_value_fe = e(count)

	reg `tabvar' L07 kabu_* if survey_round == 2, vce(cluster kecamatan)
	local p_fe = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))
	matrix p_value_fe[1,1] = `p_fe'

	eststo `row': reg `tabvar' L07 if survey_round == 2, vce(cluster kecamatan) 
	local p = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))

	matrix p_value[1,1] = `p'
	estadd matrix p_value
	estadd matrix p_value_fe

	summ `tabvar' if survey_round == 2
	matrix total_count[1,1] = round(r(mean), .001)
	estadd matrix total_count 

	summ `tabvar' if survey_round == 2 & L07 == 0
	matrix control_count[1,1] = round(r(mean), .001)
	estadd matrix control_count 

	summ `tabvar' if survey_round == 2 & L07 == 1
	matrix treat_count[1,1] = round(r(mean), .001)
	estadd matrix treat_count

	//output to latex 
	#delimit ; 

	esttab `row' using "appendix_table5a.tex", booktabs
		label
		cells("total_count control_count treat_count p_value(fmt(3)) p_value_fe(fmt(3))") 
		unstack 
		nonotes 
		nonumber 
		nomtitle 
		mlabels(none)
		collabels(none) 
		eqlabels(none)
		noobs 
		varlabels(`tabvar' "`tablabel'")
		fragment
		nolines
		append;

	#delimit cr 

}









********************************************************************************

**Table 5b: Reasons for Migration, Pooled 
//only keep kids who migrated, and one observation for each 
preserve
keep if survey_round == 2 & (migrated_midline == 1 | migrated_endline == 1)

//first: counts for migrated for school 
estpost ttest migrated_for_school, by(L07)
matrix total_count = e(count)
matrix control_count = e(count)
matrix treat_count = e(count)
matrix p_value = e(count)
matrix p_value_fe = e(count)

//include control for which survey round child migrated in 
reg migrated_for_school L07 kabu_* migrated_midline, vce(cluster kecamatan)
local p_fe = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))
matrix p_value_fe[1,1] = `p_fe'

eststo row1: reg migrated_for_school L07 migrated_midline, vce(cluster kecamatan)
local p = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))

matrix p_value[1,1] = `p'
estadd matrix p_value
estadd matrix p_value_fe

summ migrated_for_school
matrix total_count[1,1] = round(r(mean), .001)
estadd matrix total_count 

summ migrated_for_school if L07 == 0
matrix control_count[1,1] = round(r(mean), .001)
estadd matrix control_count 

summ migrated_for_school if L07 == 1
matrix treat_count[1,1] = round(r(mean), .001)
estadd matrix treat_count

//output to latex 
#delimit ; 

esttab row1 using "appendix_table5b.tex", booktabs
	label
	cells("total_count control_count treat_count p_value(fmt(3)) p_value_fe(fmt(3))") 
	unstack 
	nonotes 
	nonumber 
	nomtitle 
	collabels("Overall" "Control" "Treatment" "\shortstack{Control - Treatment \\ \emph{p}-value}" "\shortstack{Control - Treatment \\ \emph{p}-value (District FE)}", lhs("Outcome:")) 
	noobs 
	varlabels(migrated_for_school "Migrated for School")
	fragment
	replace;

#delimit cr 

//loop over other variables of interest 
forvalues i = 1/3 {

	local tabvar: word `i' of "migrated_for_work" "migrated_for_marriage" "migrated_for_other"

	local tablabel: word `i' of "Migrated for Work" "Migrated to Follow Spouse" "Migrated for Other Reason" 

	local row: word `i' of "A" "B" "C"


	estpost ttest `tabvar', by(L07)
	matrix total_count = e(count)
	matrix control_count = e(count)
	matrix treat_count = e(count)
	matrix p_value = e(count)
	matrix p_value_fe = e(count)

	reg `tabvar' L07 kabu_* migrated_midline, vce(cluster kecamatan)
	local p_fe = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))
	matrix p_value_fe[1,1] = `p_fe'

	eststo `row': reg `tabvar' L07 migrated_midline, vce(cluster kecamatan) 
	local p = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))

	matrix p_value[1,1] = `p'
	estadd matrix p_value
	estadd matrix p_value_fe

	summ `tabvar'
	matrix total_count[1,1] = round(r(mean), .001)
	estadd matrix total_count 

	summ `tabvar' if L07 == 0
	matrix control_count[1,1] = round(r(mean), .001)'
	estadd matrix control_count 

	summ `tabvar' if L07 == 1
	matrix treat_count[1,1] = round(r(mean), .001)
	estadd matrix treat_count

	//output to latex 
	cd "`latex'"
	
	#delimit ; 

	esttab `row' using "appendix_table5b.tex", booktabs
		label
		cells("total_count control_count treat_count p_value(fmt(3)) p_value_fe(fmt(3))") 
		unstack 
		nonotes 
		nonumber 
		nomtitle 
		mlabels(none)
		collabels(none) 
		eqlabels(none)
		noobs 
		varlabels(`tabvar' "`tablabel'")
		fragment
		nolines
		append;

	#delimit cr 

}

restore
