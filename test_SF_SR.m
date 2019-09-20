clear all;

%%%====== Settings ======%%%
scale = 2; % options: 2, 3, 4
gpuDevice(2);
%%%======================%%%
addpath('utils');
disp(['Testing for scale ', num2str(scale), '...'])

% load SF network
disp('Loading net...')
netstruct = load('./net/SF_subnet.mat');
net = dagnn.DagNN.loadobj(netstruct.net);
move(net,'gpu');
net.mode = 'test' ;
net.conserveMemory = true;
pred_index = net.getVarIndex('pred');

% load video SR network
disp('Loading net...')
netstruct_SR = load(sprintf('./net/x%d.mat', scale));
net_SR = dagnn.DagNN.loadobj(netstruct_SR.net);
move(net_SR,'gpu');
net_SR.mode = 'test' ;
net_SR.conserveMemory = true;
pred_index_SR = net_SR.getVarIndex('pred');

% test
disp('Testing starts...')
for data_folder = 1:5
    filelist = dir(fullfile('./data/SF_subnet', num2str(data_folder), '*.png'));
    result_dir = fullfile('./pred/SF_SR', strcat('x', num2str(scale)), num2str(data_folder));
    if ~exist(result_dir)
        mkdir(result_dir)
    end
    for i = 3:size(filelist, 1)-2
        % read 5 consecutive frames
        img_1 = rgb2ycbcr(imread(fullfile(filelist(i-2).folder, filelist(i-2).name)));
        img_2 = rgb2ycbcr(imread(fullfile(filelist(i-1).folder, filelist(i-1).name)));
        img_3 = rgb2ycbcr(imread(fullfile(filelist(i).folder, filelist(i).name)));
        img_4 = rgb2ycbcr(imread(fullfile(filelist(i+1).folder, filelist(i+1).name)));
        img_5 = rgb2ycbcr(imread(fullfile(filelist(i+2).folder, filelist(i+2).name)));
        % concatenate
        img_seq = cat(3, img_1(:, :, 1), img_2(:, :, 1), img_3(:, :, 1), img_4(:, :, 1), img_5(:, :, 1)); % [H, W, C]
        % normalize
        img_seq = single(img_seq)/255;

        %%%=========== SF Subnet ===========%%%
        % create LR data
        data_seq = imresize(img_seq, [27, 48]);
        % take Y-channel Y & change type
        data_seq = gpuArray(data_seq);
        % prediction
        net.eval({'input', data_seq});
        pred_SF = gather(net.vars(pred_index).value);
        [~, index(i)] = max(squeeze(pred_SF));
        % frame replacement pre-processing
        if index(i)<3
            disp(['window #', num2str(i-2), ': ', 'scene change occurred after ', num2str(index(i)), '-th frame'])
            new_img_seq = repmat(img_seq(:, :, index(i)+1), 1, 1, 5);
            new_img_seq(:, :, index(i)+1:end) = img_seq(:, :, index(i)+1:end);
            disp('different scene frames were replaced!')
        elseif index(i)>=3 && index(i)<5
            disp(['window #', num2str(i-2), ': ', 'scene change occurred after ', num2str(index(i)), '-th frame'])
            new_img_seq = repmat(img_seq(:, :, index(i)), 1, 1, 5);
            new_img_seq(:, :, 1:index(i)) = img_seq(:, :, 1:index(i));
            disp('different scene frames were replaced!')
        elseif index(i)==5
            disp(['window #', num2str(i-2), ': ', 'no scene change']);
            new_img_seq = img_seq;
        end

        %%%=========== Video SR Subnet ===========%%%
        %%% with SF subnet
        new_img_seq = gpuArray(new_img_seq);
        % prediction
        net_SR.eval({'input', new_img_seq});
        pred_SR = gather(net_SR.vars(pred_index_SR).value);
        % pixel shuffle
        SR = pixel_shuffle(pred_SR, scale);
        SR_YUV(:, :, 1) = uint8(SR*255);
        SR_YUV(:, :, 2:3) = imresize(img_3(:, :, 2:3), scale);
        SR_RGB = ycbcr2rgb(SR_YUV);
        imwrite(SR_RGB, fullfile(result_dir, sprintf('%d_SF.png', i-2)));
        %%% without SF subnet (for comparison)
        img_seq = gpuArray(img_seq);
        % prediction
        net_SR.eval({'input', img_seq});
        pred_SR = gather(net_SR.vars(pred_index_SR).value);
        % pixel shuffle
        SR = pixel_shuffle(pred_SR, scale);
        SR_YUV(:, :, 1) = uint8(SR*255);
        SR_YUV(:, :, 2:3) = imresize(img_3(:, :, 2:3), scale);
        SR_RGB = ycbcr2rgb(SR_YUV);
        imwrite(SR_RGB, fullfile(result_dir, sprintf('%d_noSF.png', i-2)));
    end
end