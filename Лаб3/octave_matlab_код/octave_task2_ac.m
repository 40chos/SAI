% ЛР №3. Завдання №2. Нечітка модель керування кондиціонером повітря.
% Виконано засобами Octave fuzzy-logic-toolkit (аналог MATLAB Fuzzy Logic Toolbox).

pkg load fuzzy-logic-toolkit

fis = newfis('air_conditioner');

fis = addvar(fis, 'input', 'temperature', [10 35]);
fis = addmf(fis, 'input', 1, 'дуже-холодна', 'trapmf', [9 10 14 18]);
fis = addmf(fis, 'input', 1, 'холодна', 'trimf', [15 19 23]);
fis = addmf(fis, 'input', 1, 'в-нормі', 'trimf', [20 23 26]);
fis = addmf(fis, 'input', 1, 'тепла', 'trimf', [24 27 30]);
fis = addmf(fis, 'input', 1, 'дуже-тепла', 'trapmf', [27 31 35 36]);

fis = addvar(fis, 'input', 'rate', [-2 2]);
fis = addmf(fis, 'input', 2, 'вiдємна', 'trapmf', [-2.1 -2 -0.6 -0.1]);
fis = addmf(fis, 'input', 2, 'нуль', 'trimf', [-0.3 0 0.3]);
fis = addmf(fis, 'input', 2, 'додатня', 'trapmf', [0.1 0.6 2 2.1]);

fis = addvar(fis, 'output', 'regulator', [-90 90]);
fis = addmf(fis, 'output', 1, 'великий-вліво', 'trapmf', [-91 -90 -70 -45]);
fis = addmf(fis, 'output', 1, 'невеликий-вліво', 'trimf', [-30 -12 0]);
fis = addmf(fis, 'output', 1, 'нуль', 'trimf', [-10 0 10]);
fis = addmf(fis, 'output', 1, 'невеликий-вправо', 'trimf', [0 12 30]);
fis = addmf(fis, 'output', 1, 'великий-вправо', 'trapmf', [45 70 90 91]);

% temperature terms: 1 дуже-холодна,2 холодна,3 в-нормі,4 тепла,5 дуже-тепла
% rate terms: 1 вiдємна,2 нуль,3 додатня
% output terms: 1 великий-вліво,2 невеликий-вліво,3 нуль,4 невеликий-вправо,5 великий-вправо
% правило 7 виправлено на "великий-вправо" (друкарська помилка в завданні)
ruleList = [
  5 3 1 1 1;   % 1
  5 1 2 1 1;   % 2
  4 3 1 1 1;   % 3
  4 1 3 1 1;   % 4
  1 1 5 1 1;   % 5
  1 3 4 1 1;   % 6
  2 1 5 1 1;   % 7 (виправлено)
  2 3 3 1 1;   % 8
  5 2 1 1 1;   % 9
  4 2 2 1 1;   % 10
  1 2 5 1 1;   % 11
  2 2 4 1 1;   % 12
  3 3 2 1 1;   % 13
  3 1 4 1 1;   % 14
  3 2 3 1 1    % 15
];
fis = addrule(fis, ruleList);

writefis(fis, 'air_conditioner.fis');
disp('Збережено air_conditioner.fis');

test_points = [33 1.0; 33 -1.0; 12 -1.0; 12 1.0; 23 0.0; 28 0.8; 18 -0.8];
descr = {'дуже тепло, зростає', 'дуже тепло, спадає', 'дуже холодно, спадає', ...
         'дуже холодно, зростає', 'норма, стабільно', 'тепло, зростає', 'холодно, спадає'};
printf('\n=== Перевірка нечіткого виводу (air_conditioner.fis) ===\n');
for i = 1:rows(test_points)
  out = evalfis(test_points(i, :), fis);
  printf('t=%.1f, dt=%.1f (%s) -> regulator=%.2f\n', ...
         test_points(i,1), test_points(i,2), descr{i}, out);
end
