function[] = write(class, features)
colname={'二阶矩','对比度','相关度','熵', '方差', '逆差矩'};
xlswrite(['.\features\', class, '.xlsx'] , colname, 'sheet1', 'A1');
xlswrite(['.\features\', class, '.xlsx'] , features, 'sheet1', 'A2');
end