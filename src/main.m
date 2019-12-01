clc;clear all;close all;

% class = 'scratches';
% file_dir = ['E:\pattern\dataset\', class, '\'];
% file_bmp = ls(strcat(file_dir, '/*.bmp'));
% files = cellstr(strcat(file_dir, file_bmp));
% file_num =  length(files);
% 
% gray_level = 256;
% gray_level_com = 16;
% 
% features = zeros(file_num, 6);
% 
% for n = 1:file_num
%     img = imread(files{n});
%     %imshow(img);xlabel(files{n});
%     hist_img = histeq(img);
%     [L, W] = size(hist_img);
%     for i = 1 : L
%         for j = 1 : W
%             com_img(i, j) =  idivide(hist_img(i, j), gray_level_com, 'floor');
%         end
%     end
%     glg_matrix_0 = get_glgm(com_img, L, W, gray_level_com, 0);
%     glg_matrix_45 = get_glgm(com_img, L, W, gray_level_com, 45);
%     glg_matrix_90 = get_glgm(com_img, L, W, gray_level_com, 90);
%     
%     glg_matrix = (glg_matrix_0 + glg_matrix_45 + glg_matrix_90) / 3;
%     
%     glg_norm_matrix = glgm_norm(glg_matrix);
%     
%     [features(n,1), features(n,2), features(n,3), features(n,4), features(n,5), features(n,6)] = get_features(glg_norm_matrix, gray_level_com);
% end
% 
% write(class, features);
% 
all_features = load_all_features('patches', 'scratches');

vbls={'二阶矩','对比度','相关度','熵','方差','逆矩阵'};
[coeff, score, latent, tsquared, explained, mu] = pca(all_features);
% % biplot(coeff(:,1:3),'Score',score(:,1:3),'Varlabels',vbls);
% scores = xlsread("E:\pattern\features\pca.xlsx");
% biplot(coeff(:,1:3),'Score',scores(:,1:3),'Varlabels',vbls);
% write('pca', score);
% 
% re_scaling = rescaling(all_features);
% write('rescaling', re_scaling);

meas = xlsread("E:\pattern\features\pca.xlsx");
% f = figure;
% gscatter(meas(:,1), meas(:,2), meas(:, 4),'rgb','osd');

% Mdl = fitcknn(X, Y, 'NumNeighbors', 4);
% features = meas(:,1:2);
% classlabel=meas(:,4);
% n = randperm(size(features,1));
% 
% train_features = features(n(1: 70), :);
% train_label = classlabel(n(1 : 70), :);
% 
% test_features = features(n(71: end), :);
% test_label = classlabel(n(71 : end), :);
% 
% model = classRF_train(train_features,train_label);
% 
% [T_sim,votes] = classRF_predict(test_features,model);
% index = find(T_sim ~= test_label);
% accuracy = 1-length(index)/length(test_label);

% [Train_features,PS] = mapminmax(train_features');
%  Train_features = Train_features'; 
%  Test_features = mapminmax('apply',test_features',PS); 
%  Test_features = Test_features';
%  
%  
% model = svmtrain(train_label,Train_features);
% 
% [predict_train_label] = svmpredict(train_label,Train_features,model);
% [predict_test_label] = svmpredict(test_label,Test_features,model);
% 
% compare_train = (train_label == predict_train_label);
% accuracy_train = sum(compare_train)/size(train_label,1)*100; 
% fprintf('训练集准确率：%f\n',accuracy_train)
% compare_test = (test_label == predict_test_label);
% accuracy_test = sum(compare_test)/size(test_label,1)*100;
% fprintf('测试集准确率：%f\n',accuracy_test)

data = xlsread(".\features\pca.xlsx", 'sheet1', 'A1:F100');
knn(data);








