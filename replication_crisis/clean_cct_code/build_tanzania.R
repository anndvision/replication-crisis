library(tidyverse)
rm(list = ls())
tanzania_list <- readRDS("~/Documents/conditional_cash_transfers/cct/tanzania_list.rds")
tanzania_all_df <-  tanzania_list[[1]]

##### house_tanzania_r1 ##### 
house_tanzania_r1 <- tanzania_all_df$r1_HHData %>% 
  mutate(HHNum = as.integer(HHNum) ) %>% 
  mutate(T5AQ1 = factor(T5AQ1, levels = c(1, 2, 3, 4, 5, 99),
                        labels = c("Mud/earth", "Wood/plank", "Tiles",
                                   "Concrete/Cement", "Grass", "Other (specify)"))) %>% 
  mutate(T5AQ2 = factor(T5AQ2, levels = c(1, 2, 3, 4, 5, 6, 7, 99),
                        labels = c("Mud", "Thatch", "Wood", "Iron sheets",
                                   "Concrete/Cement", "Roofing tiles", "Asbestos", "Other (specify)"))) %>% 
  mutate(T5AQ3 = factor(T5AQ3, levels = c(1, 2, 3, 4, 5, 6, 7, 99),
                        labels = c("Mud/Mud brick", "Stone", "Burnt bricks",
                                   "Concrete/Cement", "Wood/Bamboo", "Iron sheets",
                                   "Cardboard", "Other (specify)"))) %>% 
  mutate(T5AQ6 = factor(T5AQ6, levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99),
                        labels = c("Pipe bourne water treated", "Piped bourne water untreated",
                                   "Bore hole/hand pump", "Covered Well", "Uncovered Well",
                                   "Protected spring", "Unprotected spring", "Rain water",
                                   "River, lake, pond", "Truck, vendor", "Other (specify)"))) %>% 
  mutate(T5AQ7 = factor(T5AQ7, levels = c(1, 2, 3, 4, 5, 6, 7, 99),
                        labels = c("None (bush)", "Flush to sewer", "Flush to septic tank",
                                   "Pan/bucket", "Covered pit latrine", "Uncovered pit latrine",
                                   "Ventilated pit latrine", "Other (specify)"))) %>% 
  mutate(T5AQ8 = factor(T5AQ8, levels = c(1, 2, 3, 4, 5, 6, 7, 99),
                        labels = c("Kerosine/paraffin", "Gas", "Main electricity",
                                   "Solar panels/private generator", "Battery", "Candles",
                                   "Firewood", "Other (specify)"))) %>% 
  select(CLID_anonymous,HHID=HHNum,floor_type=T5AQ1, roof_type=T5AQ2,wall_type= T5AQ3,water_type= T5AQ6, toilet_type = T5AQ7, lighting_type = T5AQ8 )


  
##### household_treatment_r1 ##### 
household_treatment_r1 <- tanzania_all_df$r1_HHrespondents %>% 
  select(districtname, CLID_anonymous=clustername_anonymous, wardname, HHID=hhid, Treatment) %>% 
  mutate(Treatment = if_else(Treatment == 2, 0, Treatment) ) 
  
##### village_treatment_r1 ##### 
village_treatment_r1 <- tanzania_all_df$r1_CSCVillage %>% 
  select(-clustername, CLID_anonymous = clustername_anonymous) %>% 
  mutate(Treatment = if_else(Treatment == 2, 0, Treatment) ) 
##### member_r1 ##### 
member_r1 <- tanzania_all_df$r1_HHMember %>% 
  select(CLID_anonymous, HHNum, T3Q1, T3Q3,T3Q4, T3AQ1, T3AQ6, T3BQ4) %>% 
  mutate(HHNum = as.integer(HHNum) ) %>% 
  rename(age_r1 = T3Q4) %>% 
  mutate(T3Q3 = if_else(T3Q3 == 2, 0, T3Q3) ) %>% 
  # mutate(T3Q3 = factor(T3Q3, levels = c(1, 0),
  #                                          labels = c("Male", "Female"))) %>% 
  mutate(T3AQ1 = factor(T3AQ1, levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 99),
                        labels = c("Head", "Wife/Husband", "Biological child", "Adopted child",
                                   "Grandchild", "Niece/Nephew", "Father/Mother", "Sister/Brother",
                                   "Uncle/Aunt", "Son/Daughter in-law", "Brother/Sister in-law",
                                   "Grandfather/mother", "Father/Mother in-law", "Other relative",
                                   "Servant/servant's relative", "Lodger/lodger's relative",
                                   "Non-relative", "Other (specify)"))) %>% 
  mutate(T3AQ6 = factor(T3AQ6, levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 88, 99),
                        labels = c("Paid employee", "Agricultural sector: Self-employed WITH employees",
                                   "Agricultural: self-employed WITHOUT employees",
                                   "Non-agricultural: Self-employed WITH employees",
                                   "Non-agricultural: self-employed WITHOUT employees",
                                   "Other unpaid family work", "Domestic work", "Seeking work",
                                   "Sick", "Retired", "Full-time student", "Apprentice", "Incapacitated",
                                   "Religious leader/Pastor", "Child Care Activities", "Caring for elderly",
                                   "Casual Labourer", "DK", "Other (specify)"))) %>% 
  mutate(T3BQ4 = factor(T3BQ4, levels = as.integer(c("00", "01", "02", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                                          "20", "21", "22", "23", "24", "25", "26", "27", "28", "41", "42", "43",
                                          "44", "45", "88")),
                        labels = c("None", "Pre-Primary", "Adult", "Standard I", "Standard II", "Standard III",
                                   "Standard IV", "Standard V", "Standard VI", "Standard VII", "Standard VIII",
                                   "Primary + Course", "Form I", "Form II", "Form III", "Form IV", "Form IV + Course",
                                   "Form V", "Form VI", "Form VI+ Course", "Ordinary Diploma", "University I",
                                   "University II", "University III", "University IV", "University V & +", "DK"))) %>% 
  select(CLID_anonymous, HHID=HHNum,HHMemberID= T3Q1, age_r1, gender_r1=T3Q3, relation_head_r1=T3AQ1, main_daily_activity_r1=T3AQ6, highest_ed_member_r1 =T3BQ4)

##### household_r3 ##### 

household_r3 <- tanzania_all_df$r3_HHData %>% 
  mutate(PayConditions = factor(PayConditions, levels = c(1:7, -96),
                                labels = c("Children have to attend school", 
                                           "Children have to make monthly visit to clinic",
                                           "Elderly have to visit clinic at least once per year",
                                           "Mentioned 1 and 2",
                                           "Mentioned 2 and 3",
                                           "Mentioned 1 and 3",
                                           "Mentioned all 1, 2, and 3",
                                           "Others (specify)"))) %>% 
  mutate(NumPay = if_else(NumPay == -99, NA, NumPay)) %>% 
  # group_by(CLID_anonymous, HHID) %>% 
  # mutate(NumPay = mean(NumPay, na.rm = TRUE), 
  #        PayConditions = paste(PayConditions, collapse = "; ")) %>% 
  select(CLID_anonymous, HHID,SplitOffID, NumPay, PayConditions) # %>%
  # distinct() %>% 
  # ungroup()


##### outcomes_r3  ##### 

outcomes_r3 <- tanzania_all_df$r3_HHMember %>% 
  mutate(T3BQ1 = if_else(T3BQ1 == 2, 0, T3BQ1) ) %>% 
  # mutate(T3BQ1 = factor(T3BQ1, levels = c(1, 0),
  #                                    labels = c("Yes", "No"))) %>% 
  mutate(T3BQ7 = if_else(T3BQ7 == 2, 0, T3BQ7) ) %>% 
  # mutate(T3BQ7 = factor(T3BQ7, levels = c(1, 0),
  #                                          labels = c("Yes", "No"))) %>%
  mutate(T3BQ8 = if_else(T3BQ8 == 2, 0, T3BQ8) ) %>% 
  # mutate(school_last12_outcome = factor(T3BQ8, levels = c(1, 0),
  #                                       labels = c("Yes", "No"))) %>% 
  mutate(RelationHead = factor(RelationHead, levels = c(1:18, -96),
                                   labels = c("Head", "Wife/Husband", "Biological child", "Adopted child",
                                              "Grandchild", "Niece/Nephew", "Father/Mother", "Sister/Brother",
                                              "Uncle/Aunt", "Son/Daughter in-law", "Brother/Sister in-law",
                                              "Grandfather/mother", "Father/Mother in-law", "Other relative",
                                              "Servant/servant's relative", "Lodger/lodger's relative",
                                              "Non-relative", "Step child", "Other (specify)"))) %>% 
  mutate(Sex = if_else(Sex == 2, 0, Sex) ) %>% 
  mutate(T3BQ4 = factor(T3BQ4, levels = c(0, 1, 2, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 41, 42, 43, 44, 45, -99),
                          labels = c("None", "Pre-Primary", "Adult", "Standard I", "Standard II", "Standard III",
                                     "Standard IV", "Standard V", "Standard VI", "Standard VII", "Standard VIII",
                                     "Primary + Course", "Form I", "Form II", "Form III", "Form IV", "Form IV + Course",
                                     "Form V", "Form VI", "Form VI+ Course", "Ordinary Diploma", "University I",
                                     "University II", "University III", "University IV", "University V & +", "Do not know"))) %>% 
  select(DistID, CLID_anonymous,HHID, HHMemberID,SplitOffID, age_r3=AgeYears, gender_r3= Sex,relation_head_r3= RelationHead, RelationHead_other, read_write_outcome = T3BQ1, currently_school_outcome= T3BQ7, school_last12_outcome=T3BQ8,highest_ed_outcome = T3BQ4 )

##### link  ##### 

link <- tanzania_all_df$r3_Link_Ind %>% 
  select(BCLID_anonymous, BHHNum, BHHMemberID=BT3Q1, ECLID_anonymous, EHHID, EHHMemberID, ESplitOffID) %>% 
  mutate(BHHNum = as.integer(BHHNum))

##### data_r1  ##### 

data_r1 <- member_r1 %>% 
  left_join(household_treatment_r1) %>% 
  left_join(house_tanzania_r1) %>%
  left_join(village_treatment_r1) %>% 
  rename(BCLID_anonymous=CLID_anonymous, BHHNum=HHID,BHHMemberID=HHMemberID ) %>% 
  left_join(link )
  
##### data_r3  ##### 

data_r3 <- outcomes_r3 %>% 
  left_join(household_r3) %>% 
  rename(ECLID_anonymous=CLID_anonymous, EHHID=HHID,EHHMemberID=HHMemberID, ESplitOffID=SplitOffID) %>% 
  left_join(link )


##### final Tanzania  ##### 


final_tanzania <- data_r1 %>% 
  left_join(data_r3) 
    


children_tanzania <- final_tanzania %>% 
  filter(age_r1 < 22, age_r1 > 13) %>% 
  filter(!is.na(ECLID_anonymous))


saveRDS(final_tanzania, "final_tanzania.rds")


