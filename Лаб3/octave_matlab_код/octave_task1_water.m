% ЛР №3. Завдання №1. Нечітка модель керування кранами гарячої та холодної
% води. Виконано засобами Octave fuzzy-logic-toolkit (аналог MATLAB
% Fuzzy Logic Toolbox).

pkg load fuzzy-logic-toolkit

fis = newfis('water_mixer');

% --- Вхідні змінні ---
fis = addvar(fis, 'input', 'temperature', [0 100]);
fis = addmf(fis, 'input', 1, 'холодна', 'trapmf', [-1 0 10 30]);
fis = addmf(fis, 'input', 1, 'прохолодна', 'trimf', [15 30 45]);
fis = addmf(fis, 'input', 1, 'тепла', 'trimf', [35 50 65]);
fis = addmf(fis, 'input', 1, 'не-дуже-гаряча', 'trimf', [55 70 85]);
fis = addmf(fis, 'input', 1, 'гаряча', 'trapmf', [75 90 100 101]);

fis = addvar(fis, 'input', 'pressure', [0 10]);
fis = addmf(fis, 'input', 2, 'слабкий', 'trapmf', [-1 0 2 5]);
fis = addmf(fis, 'input', 2, 'не-дуже-сильний', 'trimf', [3 5 7]);
fis = addmf(fis, 'input', 2, 'сильний', 'trapmf', [5 8 10 11]);

% --- Вихідні змінні (кут крана гарячої та холодної води) ---
angle_terms = {'великий-вліво', 'середній-вліво', 'невеликий-вліво', ...
               'нуль', 'невеликий-вправо', 'середній-вправо', 'великий-вправо'};
angle_params = {[-91 -90 -70 -45], [-60 -40 -15], [-25 -12 0], [-10 0 10], ...
                [0 12 25], [15 40 60], [45 70 90 91]};
angle_types = {'trapmf','trimf','trimf','trimf','trimf','trimf','trapmf'};

fis = addvar(fis, 'output', 'hot_angle', [-90 90]);
fis = addvar(fis, 'output', 'cold_angle', [-90 90]);
for k = 1:numel(angle_terms)
  fis = addmf(fis, 'output', 1, angle_terms{k}, angle_types{k}, angle_params{k});
  fis = addmf(fis, 'output', 2, angle_terms{k}, angle_types{k}, angle_params{k});
end

% Індекси термів для зручності формування правил:
% temperature: 1 холодна,2 прохолодна,3 тепла,4 не-дуже-гаряча,5 гаряча
% pressure: 1 слабкий,2 не-дуже-сильний,3 сильний
% output terms: 1 великий-вліво,2 середній-вліво,3 невеликий-вліво,4 нуль,
%               5 невеликий-вправо,6 середній-вправо,7 великий-вправо

% [temperature pressure hot_angle cold_angle weight connective(1=and)]
ruleList = [
  5 3 2 6 1 1;   % 1. гаряча & сильний -> hot=серед.вліво, cold=серед.вправо
  5 2 4 6 1 1;   % 2. гаряча & не дуже сильний -> cold=серед.вправо
  4 3 3 4 1 1;   % 3. не дуже гаряча & сильний -> hot=невел.вліво
  4 1 5 5 1 1;   % 4. не дуже гаряча & слабкий -> hot=невел.вправо, cold=невел.вправо
  3 2 4 4 1 1;   % 5. тепла & не дуже сильний -> (0,0)
  2 3 6 2 1 1;   % 6. прохолодна & сильний -> hot=серед.вправо, cold=серед.вліво
  2 2 6 3 1 1;   % 7. прохолодна & не дуже сильний -> hot=серед.вправо, cold=невел.вліво
  1 1 7 4 1 1;   % 8. холодна & слабкий -> hot=великий вправо
  1 3 2 6 1 1;   % 9. холодна & сильний -> hot=серед.вліво, cold=серед.вправо
  3 3 3 3 1 1;   % 10. тепла & сильний -> hot=невел.вліво, cold=невел.вліво
  3 1 5 5 1 1    % 11. тепла & слабкий -> hot=невел.вправо, cold=невел.вправо
];
fis = addrule(fis, ruleList);

writefis(fis, 'water_mixer.fis');
disp('Збережено water_mixer.fis');

% --- Перевірка для тестових точок ---
test_points = [95 9; 50 5; 20 9; 5 1; 30 4];
descr = {'дуже гаряча, сильний напір', 'тепла, середній напір', ...
         'холодна, сильний напір', 'дуже холодна, слабкий напір', ...
         'прохолодна, помірний напір'};
printf('\n=== Перевірка нечіткого виводу (water_mixer.fis) ===\n');
for i = 1:rows(test_points)
  out = evalfis(test_points(i, :), fis);
  printf('T=%d, P=%d (%s) -> hot=%.2f, cold=%.2f\n', ...
         test_points(i,1), test_points(i,2), descr{i}, out(1), out(2));
end
