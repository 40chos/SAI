% ЛР №3, Завдання 1 (приклад). Система нечіткого виводу "first"
% для функції y=(x1^2-8)*cos(x2), x1,x2 є [0,4].
% Виконано в GNU Octave (пакет fuzzy-logic-toolkit).

pkg load fuzzy-logic-toolkit
graphics_toolkit("gnuplot")

% --- Побудова графіка цільової функції (Рис. 1 методички) ---
n = 15;
x1 = 0:4/(n-1):4;
x2 = 0:4/(n-1):4;
y = zeros(n, n);
for j = 1:n
  y(j, :) = (x1.^2 - 8) * cos(x2(j));
end
figure(1, "visible", "off");
surf(x1, x2, y);
xlabel('x1'); ylabel('x2'); zlabel('y'); title('Target');
print(1, 'octave_fig1_target.png', '-dpng');

% --- Крок 1-6: створення FIS, перейменування змінних ---
fis = newfis('first');
fis = addvar(fis, 'input', 'x1', [0 4]);
fis = addvar(fis, 'input', 'x2', [0 4]);
fis = addvar(fis, 'output', 'y', [-10 10]);

% --- Крок 9-10: x1 - 3 трикутних терми ---
fis = addmf(fis, 'input', 1, 'L', 'trimf', [-2 0 2]);
fis = addmf(fis, 'input', 1, 'A', 'trimf', [0 2 4]);
fis = addmf(fis, 'input', 1, 'H', 'trimf', [2 4 6]);

% --- Крок 11-12: x2 - 5 гаусових термів ---
fis = addmf(fis, 'input', 2, 'L',  'gaussmf', [0.4247 0]);
fis = addmf(fis, 'input', 2, 'LA', 'gaussmf', [0.4247 1]);
fis = addmf(fis, 'input', 2, 'A',  'gaussmf', [0.4247 2]);
fis = addmf(fis, 'input', 2, 'HA', 'gaussmf', [0.4247 3]);
fis = addmf(fis, 'input', 2, 'H',  'gaussmf', [0.4247 4]);

% --- Крок 13-14: y - 5 трикутних термів ---
fis = addmf(fis, 'output', 1, 'L',  'trimf', [-15 -10 -5]);
fis = addmf(fis, 'output', 1, 'LA', 'trimf', [-10 -5 0]);
fis = addmf(fis, 'output', 1, 'A',  'trimf', [-5 0 5]);
fis = addmf(fis, 'output', 1, 'HA', 'trimf', [0 5 10]);
fis = addmf(fis, 'output', 1, 'H',  'trimf', [5 10 15]);

% --- Крок 16: 9 правил бази знань ---
% [x1 x2 y вага зв'язка(1=and)]
ruleList = [
  1 1 1 1 1;
  1 5 3 1 1;
  1 4 5 1 1;
  3 1 4 1 1;
  3 5 1 1 1;
  2 3 3 1 1;
  2 4 4 1 1;
  1 2 2 1 1;
  2 4 3 1 1
];
fis = addrule(fis, ruleList);

% --- Збереження FIS у файл (Крок 6/17: Export to disk) ---
writefis(fis, 'first.fis');
disp('Систему збережено у файл first.fis');

% --- Перевірка нечіткого виводу для x1=2, x2=2 (аналог Рис. 6) ---
result = evalfis([2 2], fis);
printf('Перевірка (аналог Рис. 6): x1=2, x2=2 -> y = %.6f\n', result);

% --- Побудова поверхні системи нечіткого виводу (аналог Рис. 7, View surface) ---
n2 = 15;
x1g = linspace(0, 4, n2);
x2g = linspace(0, 4, n2);
Yfis = zeros(n2, n2);
for i = 1:n2
  for j = 1:n2
    Yfis(j, i) = evalfis([x1g(i) x2g(j)], fis);
  end
end
figure(2, "visible", "off");
surf(x1g, x2g, Yfis);
xlabel('x1'); ylabel('x2'); zlabel('y');
title('Fuzzy system surface (first.fis)');
print(2, 'octave_fig2_fis_surface.png', '-dpng');

disp('Готово. Графіки: octave_fig1_target.png, octave_fig2_fis_surface.png');
