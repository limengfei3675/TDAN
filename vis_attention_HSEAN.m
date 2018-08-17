% clear; clc;

TEST_NUM = 20; % Number of test 
QUES_ID_START = 200; % Start of testing question id

image_path = '/disk1/hbliu/mscoco/test2015/';  % Image file

att_h5 = '/home/lmf/mcan_for_vqa2/predict/test_vis_ques_att_box.h5'; % Attention map file
% ques_json = 'test_prepro_questions.json'; % Question file
% { 4195880:{'question': u'Are the dogs tied?', 'image_name': u'COCO_test2015_000000419588.jpg'}, 
% 4195881 {'question': u'Is this a car show?', 'image_name': u'COCO_test2015_000000419588.jpg'}, }


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read related files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Question file
% if ~exist('jsonData','var')
%     addpath('jsonlab-1.2');
%     jsonData = loadjson(ques_json);
%     raw_que_list = fieldnames(jsonData);    % question id list
% end

% que_num = length(raw_que_list)  % # of Questions: 60864
% que_num = TEST_NUM; % number of examples to show
% que_id1 = QUES_ID_START % Start of testing question id
imdir='COCO_test2015_%012d.jpg'
name = sprintf(imdir, 10);

%  f.create_dataset("vis1", data=vis1)
%             f.create_dataset("vis2", data=vis2)
%             f.create_dataset("pure1", data=pure1)
%             f.create_dataset("pure2", data=pure2)
%             f.create_dataset("image_id",data=images_id)
%             f.create_dataset("ques_id", data=questions_id)
%             f.create_dataset("boxes", data=bbox)
% Attention file
qids_data = h5read(att_h5, '/ques_id');
imgids_data = hdf5read(att_h5, '/image_id');
pure1_data = hdf5read(att_h5, '/pure1');
att1_data = hdf5read(att_h5, '/vis1');
pure2_data = hdf5read(att_h5, '/pure2');
att2_data = hdf5read(att_h5, '/vis2');
size(att1_data);
boxex_data =  hdf5read(att_h5, '/boxes');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize attention maps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i= 1:10000
    fprintf('This is the sample %d.\n\n', i)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Obtain question string, image name
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    que_id = qids_data(i);
    %que_id = strcat(raw_que_id(5), raw_que_id(7:end));  % question id
    fprintf('question id is: %s \n', que_id);
    img_id = imgids_data(i);
   % que_str = que_struct.question;  % question string
     % image name
    img_str = sprintf(imdir, img_id);
%     fprintf('question is: %s \n', que_str);
    fprintf('image name is: %s \n\n',  img_str); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % New firgure
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %idx = find(qids_data==str2num(que_id))
   % qids_data(idx)
    
    img_path = strcat(image_path, img_str);
    ori_img = imread(img_path); % load image
    h = size(ori_img,1);
    w = size(ori_img,2);

    figure; % new firgure window
    set(gcf, 'position', [-1000 500 900 500]); %[left bottom width height]
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Visualize attention map for branch 2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(1,2,2)
    imshow(ori_img);    % display the original image
    hold on;
    
%     box_data = hdf5read(box_h5, img_str)'; % Box: 19x4 [Xmin,Ymin,Xmax,Ymax]
    box_data = boxex_data(:,:,i); % 36 x 4 [x1,y1,x2,y2]
    att_data =  pure1_data(:,i);  % attention weight: 19
    sum_prob = sum(att_data);
    heatmap1 = zeros(w,h);
    
    for box_id = 1:36
        att = att_data(box_id);
        box = box_data(:,box_id);
        
        x_min = floor(box(1));
        if x_min == 0 
            x_min = 1;
        end
        x_max = floor(box(3));
        y_min = floor(box(2));
        if y_min == 0 
            y_min = 1;
        end
        y_max = floor(box(4));        
        heatmap1(x_min:x_max, y_min:y_max) = heatmap1(x_min:x_max, y_min:y_max) + att;
    end
    
    heatmap1 = heatmap1';
    heatmap1 = heatmap1/max(heatmap1(:))*255;
    imagesc(imresize(heatmap1,[h,w]),'AlphaData',0.5);
    
    set(gcf,'color','none');
    axis([0 w 0 h]);
    set(gca,'position',[0 0 1 1]);
    saveas(gcf,strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id), '_pure1.png')) ;
    close;
    % vis1
    figure; % new firgure window
    set(gcf, 'position', [-1000 500 900 500]); %[left bottom width height]
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Visualize attention map for branch 2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(1,2,2)
    imshow(ori_img);    % display the original image
    hold on;
    
%     box_data = hdf5read(box_h5, img_str)'; % Box: 19x4 [Xmin,Ymin,Xmax,Ymax]
    box_data = boxex_data(:,:,i); % 36 x 4 [x1,y1,x2,y2]
    att_data =  att1_data(:,i);  % attention weight: 19
    sum_prob = sum(att_data);
    heatmap2 = zeros(w,h);
    
    for box_id = 1:36
        att = att_data(box_id);
        box = box_data(:,box_id);
        
        x_min = floor(box(1));
        if x_min == 0 
            x_min = 1;
        end
        x_max = floor(box(3));
        y_min = floor(box(2));
        if y_min == 0 
            y_min = 1;
        end
        y_max = floor(box(4));
        
        heatmap2(x_min:x_max, y_min:y_max) = heatmap2(x_min:x_max, y_min:y_max) + att;
    end
    
    heatmap2 = heatmap2';
    heatmap2 = heatmap2/max(heatmap2(:))*255;
    imagesc(imresize(heatmap2,[h,w]),'AlphaData',0.5);
    
    set(gcf,'color','none');
    axis([0 w 0 h]);
    set(gca,'position',[0 0 1 1]);
    saveas(gcf,strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id), '_vis1.png')) ;
    close;
    
    % pure2
    figure; % new firgure window
    set(gcf, 'position', [-1000 500 900 500]); %[left bottom width height]
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Visualize attention map for branch 2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(1,2,2)
    imshow(ori_img);    % display the original image
    hold on;
    
%     box_data = hdf5read(box_h5, img_str)'; % Box: 19x4 [Xmin,Ymin,Xmax,Ymax]
    box_data = boxex_data(:,:,i); % 36 x 4 [x1,y1,x2,y2]
    att_data =  pure2_data(:,i);  % attention weight: 19
    sum_prob = sum(att_data);
    heatmap3 = zeros(w,h);
    
    for box_id = 1:36
        att = att_data(box_id);
        box = box_data(:,box_id);
        
        x_min = floor(box(1));
        if x_min == 0 
            x_min = 1;
        end
        x_max = floor(box(3));
        y_min = floor(box(2));
        if y_min == 0 
            y_min = 1;
        end
        y_max = floor(box(4));
        
        heatmap3(x_min:x_max, y_min:y_max) = heatmap3(x_min:x_max, y_min:y_max) + att;
    end
    
    heatmap3 = heatmap3';
    heatmap3 = heatmap3/max(heatmap3(:))*255;
    imagesc(imresize(heatmap3,[h,w]),'AlphaData',0.5);
    
    set(gcf,'color','none');
    axis([0 w 0 h]);
    set(gca,'position',[0 0 1 1]);
    saveas(gcf,strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id), '_pure2.png')) ;
    close;
    
    % vis2
    figure; % new firgure window
    set(gcf, 'position', [-1000 500 900 500]); %[left bottom width height]
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Visualize attention map for branch 2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(1,2,2)
    imshow(ori_img);    % display the original image
    hold on;
    
%     box_data = hdf5read(box_h5, img_str)'; % Box: 19x4 [Xmin,Ymin,Xmax,Ymax]
    box_data = boxex_data(:,:,i); % 36 x 4 [x1,y1,x2,y2]
    att_data =  att2_data(:,i);  % attention weight: 19
    sum_prob = sum(att_data);
    heatmap4 = zeros(w,h);
    
    for box_id = 1:36
        att = att_data(box_id);
        box = box_data(:,box_id);
        
        x_min = floor(box(1));
        if x_min == 0 
            x_min = 1;
        end
        x_max = floor(box(3));
        y_min = floor(box(2));
        if y_min == 0 
            y_min = 1;
        end
        y_max = floor(box(4));
        
        heatmap4(x_min:x_max, y_min:y_max) = heatmap4(x_min:x_max, y_min:y_max) + att;
    end
    
    heatmap4 = heatmap4';
    heatmap4 = heatmap4/max(heatmap4(:))*255;
    imagesc(imresize(heatmap4,[h,w]),'AlphaData',0.5);
    
    set(gcf,'color','none');
    axis([0 w 0 h]);
    set(gca,'position',[0 0 1 1]);
    saveas(gcf,strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id), '_vis2.png')) ;
    close;
    
end
