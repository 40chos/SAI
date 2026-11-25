pkg load fuzzy-logic-toolkit
set(0, 'DefaultFigureVisible', 'off');

% ===================== first.fis (приклад) =====================
fis = readfis('first.fis');
plotmf(fis, 'input', 1); print(gcf, 'oct_first_x1_mf.png', '-dpng'); close all;
plotmf(fis, 'input', 2); print(gcf, 'oct_first_x2_mf.png', '-dpng'); close all;
plotmf(fis, 'output', 1); print(gcf, 'oct_first_y_mf.png', '-dpng'); close all;
gensurf(fis); print(gcf, 'oct_first_gensurf.png', '-dpng'); close all;

% ===================== water_mixer.fis =====================
fis2 = readfis('water_mixer.fis');
plotmf(fis2, 'input', 1); print(gcf, 'oct_water_temp_mf.png', '-dpng'); close all;
plotmf(fis2, 'input', 2); print(gcf, 'oct_water_pressure_mf.png', '-dpng'); close all;
plotmf(fis2, 'output', 1); print(gcf, 'oct_water_hotangle_mf.png', '-dpng'); close all;
gensurf(fis2, [1 2], 1); print(gcf, 'oct_water_gensurf_hot.png', '-dpng'); close all;

% ===================== air_conditioner.fis =====================
fis3 = readfis('air_conditioner.fis');
plotmf(fis3, 'input', 1); print(gcf, 'oct_ac_temp_mf.png', '-dpng'); close all;
plotmf(fis3, 'input', 2); print(gcf, 'oct_ac_rate_mf.png', '-dpng'); close all;
plotmf(fis3, 'output', 1); print(gcf, 'oct_ac_regulator_mf.png', '-dpng'); close all;
gensurf(fis3); print(gcf, 'oct_ac_gensurf.png', '-dpng'); close all;

disp('Усі графіки збережено');
