function net = net(scale)
%%%============== Initialize ================%%%
net = dagnn.DagNN();
reluBlock = dagnn.ReLU;

ch = 64;
convBlock_input = dagnn.Conv3D('size', [3 3 3 1 ch], 'hasBias', true, 'stride', [1,1,1], 'pad', [1,1,1,1,1,1]);
convBlock_c = dagnn.Conv3D('size', [3 3 3 ch ch], 'hasBias', true, 'stride', [1,1,1], 'pad', [1,1,1,1,1,1]);
convBlock_c2 = dagnn.Conv3D('size', [3 3 3 ch ch], 'hasBias', true, 'stride', [1,1,1], 'pad', [1,1,1,1,0,0]);
convBlock_scalec = dagnn.Conv3D('size', [3 3 3 ch scale*scale], 'hasBias', true, 'stride', [1,1,1], 'pad', [1,1,1,1,0,0]);

%%%============== Model ================%%%
net.addLayer('conv1', convBlock_input, {'input'}, {'conv1'}, {'conv_1f', 'conv_1b'});
net.addLayer('relu1', reluBlock, {'conv1'}, {'conv1a'}, {});

net.addLayer('conv2', convBlock_c, {'conv1a'}, {'conv2'}, {'conv_2f', 'conv_2b'});
net.addLayer('relu2', reluBlock, {'conv2'}, {'conv2a'}, {});

net.addLayer('conv3', convBlock_c, {'conv2a'}, {'conv3'}, {'conv_3f', 'conv_3b'});
net.addLayer('relu3', reluBlock, {'conv3'}, {'conv3a'}, {});

net.addLayer('conv4', convBlock_c, {'conv3a'}, {'conv4'}, {'conv_4f', 'conv_4b'});
net.addLayer('relu4', reluBlock, {'conv4'}, {'conv4a'}, {});

net.addLayer('conv5', convBlock_c2, {'conv4a'}, {'conv5'}, {'conv_5f', 'conv_5b'});
net.addLayer('relu5', reluBlock, {'conv5'}, {'conv5a'}, {});

net.addLayer('conv6', convBlock_scalec, {'conv5a'}, {'conv6'}, {'conv_6f', 'conv_6b'});

net.addLayer('Bicubic_PS', dagnn.Bicubic_PS(), {'input'}, {'bic_im'});
net.addLayer('Sum', dagnn.Sum(), {'conv6', 'bic_im'}, {'pred'});

net.addLayer('loss', dagnn.PSNRLoss3D(), {'pred', 'label'}, 'objective');
net.initParams();