function train(varargin)
clear mex_conv3d;
% set scale
scale = 2;
% load data & label
data = load(sprintf('./data/train/LR_x%d.mat', scale));
label = load(sprintf('./data/train/HR_x%d.mat', scale));
imdb.images.data = data.LR;
imdb.images.label = label.HR;
imdb.images.set = cat(2,ones(1, size(data.LR, 5)-500), 2*ones(1, 500));

% set CNN model
network = net(scale);

% set the learning rate and weight decay for biases
% default values are used for filters
for i = 2:2:12
    network.params(i).learningRate = 0.1;
    network.params(i).weightDecay = 0;
end
network.conserveMemory = true;

% options
opts.solver=@adam;
opts.train.batchSize = 32;
opts.train.continue = false; 
opts.train.gpus = 1;
opts.train.prefetch = false ;
opts.train.expDir = sprintf('./net/net_x%d', scale); 
opts.train.learningRate = [1e-4*ones(1,700) 1e-5*ones(1,100) 1e-6*ones(1,100)];
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts.train.derOutputs = {'objective',1} ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

%record
if(~isdir(opts.expDir))
    mkdir(opts.expDir);
end

% Call training function
[network,info] = cnn_train_dag(network, imdb, @getBatch,opts) ;

function inputs = getBatch(imdb, batch,opts)
images = imdb.images.data(:,:,:,:,batch) ;
labels = imdb.images.label(:,:,:,:,batch) ;

images = single(images)/255;
labels = single(labels)/255;
inputs = {'input',gpuArray(images),'label', gpuArray(labels)} ;