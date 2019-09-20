clear all;

% load data
filelist = dir(fullfile('./data/SF_subnet', '*.png'));
% load net
disp('Loading net...')
netstruct = load('./net/SF_subnet.mat');
net = dagnn.DagNN.loadobj(netstruct.net);
move(net,'gpu');
net.mode = 'test' ;
net.conserveMemory = true;
pred_index = net.getVarIndex('pred');
% test
disp('Testing starts...')
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
    % create LR data
    data_seq = imresize(img_seq, [27, 48]);
    % take Y-channel Y & change type
    data_seq = gpuArray(data_seq);
    % prediction
    net.eval({'input', data_seq});
    pred = gather(net.vars(pred_index).value);
    [~, index(i)] = max(squeeze(pred));
    % display prediction
    if index(i)<5
        disp(['window #', num2str(i-2), ': ', 'scene change occurred after ', num2str(index(i)), '-th frame'])
    else
        disp(['window #', num2str(i-2), ': ', 'no scene change']);
    end
end