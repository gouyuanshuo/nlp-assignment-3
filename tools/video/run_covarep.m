pkg load signal;
pkg load optim;
addpath(genpath('/opt/covarep'));
rmpath('/opt/covarep/external/backcompatibility_2015');
addpath('/opt/compat');
COVAREP_feature_extraction('/work', 0.01);
if ~exist('/work/audio.mat', 'file')
    error('COVAREP did not produce its expected features');
end
