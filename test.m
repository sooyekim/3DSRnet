clear all;
%%%====== Settings ======%%%
scale = 4; % options: 2, 3, 4
sequence_name = 'walk'; % options: 'calendar', 'city', 'foliage', 'walk'
%%%======================%%%
addpath('utils');
% load HR data
disp(['Testing for scale ', num2str(scale), '...'])
disp(['Loading file: ', sequence_name])
filelist = dir(fullfile('./data/test', sequence_name, '*.png'));
sz = size(imread(fullfile(filelist(1).folder, filelist(1).name)));

% initialize
hei = floor(sz(1)/scale)*scale;
wid = floor(sz(2)/scale)*scale;

psnr_Bic = zeros(1, size(filelist, 1)-4);
psnr_SR = zeros(1, size(filelist, 1)-4);
SR = zeros(hei, wid);
result_dir = fullfile('./pred', strcat('x', num2str(scale)), sequence_name);
if ~exist(result_dir)
    mkdir(result_dir)
end

% load net
disp('Loading net...')
netstruct = load(sprintf('./net/x%d.mat', scale));
net = dagnn.DagNN.loadobj(netstruct.net);
move(net,'gpu');
net.mode = 'test' ;
net.conserveMemory = true;
pred_index = net.getVarIndex('pred');

% test
disp('Testing starts...')
for i = 3:size(filelist, 1)-2
    % read 5 consecutive HR frames
    label_1 = rgb2ycbcr(imread(fullfile(filelist(i-2).folder, filelist(i-2).name)));
    label_2 = rgb2ycbcr(imread(fullfile(filelist(i-1).folder, filelist(i-1).name)));
    label_3 = rgb2ycbcr(imread(fullfile(filelist(i).folder, filelist(i).name)));
    label_4 = rgb2ycbcr(imread(fullfile(filelist(i+1).folder, filelist(i+1).name)));
    label_5 = rgb2ycbcr(imread(fullfile(filelist(i+2).folder, filelist(i+2).name)));
    % concatenate
    label_seq = cat(4, label_1, label_2, label_3, label_4, label_5); % [H, W, C, D]
    % crop & permute
    label_seq = label_seq(1:hei, 1:wid, :, :);
    label_seq = permute(label_seq, [1, 2, 4, 3]); % [H, W, D, C]
    % normalize
    label_seq = single(label_seq)/255;
    % create LR data
    data_seq = imresize(label_seq, 1/scale);
    % take Y-channel Y & change type
    data_seq_y = gpuArray(data_seq(:, :, :, 1));
    % prediction
    net.eval({'input', data_seq_y});
    pred = gather(net.vars(pred_index).value);
    % pixel shuffle
    SR = pixel_shuffle(pred, scale);
    % create bicubic image for comparison
    Bic_YUV = imresize(squeeze(data_seq(:, :, 3, :)), scale);
    % create color SR image
    SR_YUV(:, :, 1) = SR;
    SR_YUV(:, :, 2:3) = Bic_YUV(:, :, 2:3);
    % evaluation on PSNR
    psnr_Bic(i-2) = psnr(squeeze(label_seq(:, :, 3, 1)), Bic_YUV(:, :, 1));
    psnr_SR(i-2) = psnr(squeeze(label_seq(:, :, 3, 1)), single(SR_YUV(:, :, 1)));
    % display PSNR
    disp(['#', num2str(i-2), ' Bicubic PSNR: ', num2str(psnr_Bic(i-2)), ...
        ' dB  ', 'SR PSNR: ', num2str(psnr_SR(i-2)), ' dB']);
    % save SR frame
    SR_RGB = ycbcr2rgb(uint8(SR_YUV*255));
    imwrite(SR_RGB, fullfile(result_dir, sprintf('%d.png', i-2)));
end
disp([sequence_name, ' sequence -', ' Average Bic PSNR: ', num2str(mean(psnr_Bic)), ' dB'])
disp([sequence_name, ' sequence -', ' Average SR PSNR: ', num2str(mean(psnr_SR)), ' dB'])

% REFERENCE (AVG of 4 sequences)
% x2: 32.30 dB, x3: 27.71 dB, x4: 25.72 dB