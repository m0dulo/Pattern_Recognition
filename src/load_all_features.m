function[dst] = load_all_features(class1, class2)
feature1 = xlsread(['.\features\', class1, '.xlsx'], 'sheet1', 'A2:F51');
feature2 = xlsread(['.\features\', class2, '.xlsx'], 'sheet1', 'A2:F51');

dst = [feature1; feature2];

end