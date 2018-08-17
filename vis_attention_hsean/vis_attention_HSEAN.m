% clear; clc;

TEST_NUM = 20; % Number of test 
QUES_ID_START = 200; % Start of testing question id

image_path = '/disk1/hbliu/mscoco/test2015/';  % Image file

att_h5 = '/home/lmf/mcan_for_vqa2/predict/test_dual_cross_guided_att_weights_two_layer.h5'; % Attention map file
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
%{
qids_data = h5read(att_h5, '/ques_id');
imgids_data = hdf5read(att_h5, '/image_id');
att2_data = hdf5read(att_h5, '/vis2');
vis_mfh_1 = hdf5read(att_h5, '/vis_mfh_1');
vis_mfh_2 = hdf5read(att_h5, '/vis_mfh_2');
model = hdf5read(att_h5, '/model');
top_answers = hdf5read(att_h5, '/top_answers');
top = hdf5read(att_h5, '/top');
top_answers_0 = hdf5read(att_h5, '/top_answers_0');
top_0 = hdf5read(att_h5, '/top_0');
top_answers_1 = hdf5read(att_h5, '/top_answers_1');
top_1 = hdf5read(att_h5, '/top_1');
question_str = hdf5read(att_h5, '/question_str');
model = hdf5read(att_h5, '/model');
%}
qids_data = h5read(att_h5, '/ques_id');
imgids_data = hdf5read(att_h5, '/image_id');
vis_mfh_1 = hdf5read(att_h5, '/vis_mfh');
top_answers = hdf5read(att_h5, '/top_answers');
top = hdf5read(att_h5, '/top');
question_str = hdf5read(att_h5, '/question_str');
boxex_data =  hdf5read(att_h5, '/boxes');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize attention maps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i= 1:300
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
	
	y = [top(1,i); top(2,i); top(3,i); top(4,i);top(5,i)];
	b=bar(y,0.5);
    set(gca, 'xticklabel', {top_answers(1,i).Data,top_answers(2,i).Data,top_answers(3,i).Data,top_answers(4,i).Data,top_answers(5,i).Data});
    grid on;
    xlabel('answer');  
    ylabel('Percentage');  
    title(question_str(i).Data)
    if ~exist(strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id))) 
        mkdir(strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id)))         % 若不存在，在当前目录中产生一个子目录‘Figure’
    end
    saveas(gcf,strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id),'/top_ans.png'))
    
	
	%{
    y = [top_0(1,i); top_0(2,i); top_0(3,i); top_0(4,i);top_0(5,i)];
	b=bar(y,0.5);
    set(gca, 'xticklabel', {top_answers_0(1,i).Data,top_answers_0(2,i).Data,top_answers_0(3,i).Data,top_answers_0(4,i).Data,top_answers_0(5,i).Data});
    grid on;
    xlabel('answer');  
    ylabel('Percentage');  
    title(question_str(i).Data)
    saveas(gcf,strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id),'/top_ans_0.png'))
	
	y = [top_1(1,i); top_1(2,i); top_1(3,i); top_1(4,i);top_1(5,i)];
	b=bar(y,0.5);
    set(gca, 'xticklabel', {top_answers_1(1,i).Data,top_answers_1(2,i).Data,top_answers_1(3,i).Data,top_answers_1(4,i).Data,top_answers_1(5,i).Data});
    grid on;
    xlabel('answer');  
    ylabel('Percentage');  
    title(question_str(i).Data)
    saveas(gcf,strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id),'/top_ans_1.png'))
	%}
	
	
    %{
	y = [model(1,i); model(2,i)];
	b=bar(y,0.5);
    set(gca, 'xticklabel', {'MCAN','MFH-co-attention'});
    grid on;
    xlabel('model');  
    ylabel('Percentage');  
    title(question_str(i).Data)
    saveas(gcf,strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id),'/model.png'))
	%}
	
	
	%{
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
    saveas(gcf,strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id),'/vis2.png')) ;
    close;
	%}

	% vis_mfh_1
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
    att_data =  vis_mfh_1(:,i);  % attention weight: 19
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
    saveas(gcf,strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id),'/vis_mfh_1.png')) ;
    close;
	
	
	%{
	% vis_mfh_2
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
    att_data =  vis_mfh_2(:,i);  % attention weight: 19
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
    saveas(gcf,strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id),'/vis_mfh_2.png')) ;
    close;
    %}
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
    att_data =  vis_mfh_1(:,i);  % attention weight: 19
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
        
    end
    
   
    set(gcf,'color','none');
    axis([0 w 0 h]);
    set(gca,'position',[0 0 1 1]);
    saveas(gcf,strcat('hierarchical_semantic_enhanced_vis/',int2str(img_id),'_',int2str(que_id),'/origin.png')) ;
    close;
end
