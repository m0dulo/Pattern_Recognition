function[] = write(class, features)
colname={'���׾�','�Աȶ�','��ض�','��', '����', '����'};
xlswrite(['.\features\', class, '.xlsx'] , colname, 'sheet1', 'A1');
xlswrite(['.\features\', class, '.xlsx'] , features, 'sheet1', 'A2');
end