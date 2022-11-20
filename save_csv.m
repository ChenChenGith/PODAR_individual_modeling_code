% ========================================================================= 
% Code is a part of the article:
% Which parts of the road guide obstacle avoidance? 
%      Quantifying the driver's risk field
% Sarvesh Kolekar*, Joost de Winter, David Abbink
% Department of Cognitive Robotics, Faculty of Mechanical, Maritime, and 
% Materials Engineering (3ME), Delft University of Technology� Netherlands
% *Corresponding author: 
% Sarvesh Kolekar (s.b.kolekar@tudelft.nl, kolekar.sarvesh380@gmail.com)
% =========================================================================
for id=1:8
    SNo=id; % Subject Number
    ObsNo=61; % Check the Data_extraction.png
    
    [AlyCode_dir,MainData_dir,SubjectiveData_dir,ObjectiveData_dir,...
                ExptSetup_dir] = Set_file_paths (SNo,1);
    
    % Subjective answers
    cd(AlyCode_dir)
    [sa_me_avg,Obsoffx_avg,Obsoffy_avg,sa_me_sorted,Obsoffx_sorted,...
                sorting_order,Obsoffx_resorted,Obsoffy_resorted] = ...
                Subjective_Analysis(SNo,ExptSetup_dir,SubjectiveData_dir);
            
    % Objective metrics
    % You can extract other metrics than what we have extracted
    cd(AlyCode_dir)
    [ meanST_avg,SteeringTorque_sorted,LDs,...
               SteeringAngle_sorted,SteeringVelocity_sorted,...
               SimulationTime_sorted,LateralDeviation_sorted,...
               Heading_angle_sorted]...
               = Objective_Analysis(SNo,ObjectiveData_dir,...
                                sorting_order,Obsoffx_sorted,sa_me_sorted);
    
    % plotting the extracted data 
    cd(AlyCode_dir)
    Plot_data_extraction( SNo,ObsNo,LDs,SteeringTorque_sorted,...
                    SteeringAngle_sorted,LateralDeviation_sorted,Obsoffy_resorted,...
                    Obsoffx_resorted,sa_me_sorted);

    csvwrite(['angle_P',num2str(SNo),'.csv'], SteeringAngle_sorted)
    csvwrite(['sub_resp_P',num2str(SNo),'.csv'], sa_me_avg)
end
% csvwrite('LDs.csv', LDs)
% csvwrite('Obsoffx.csv', Obsoffx_resorted)
% csvwrite('Obsoffy.csv', Obsoffy_resorted)