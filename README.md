# ANN-optimizer
Оптимизация гиперпараметров ИНН методом генетических алгоритмов.

Структура классов программной реализации алгоритма оптимизации гиперпараметров нейронных сетей:
* ANNOptimizer	Статический главный класс программы
* CompData	Класс данных для обучения и проверки функционирования сети
* Func	Статический класс с методами, не включенными в другие классы
* Gene	Класс генотипа особи генетического алгоритма
* Individ	Класс особи генетического алгоритма
* Mutation	Статический класс мутаций генетического алгоритма
* Population	Класс популяции генетического алгоритма

Главный класс включает только метод main. 
В этом методе производится импорт файлов нейронных сетей (*.nnet), контрольных и обучающих выборок (*.dset); 
выбираются величины допустимой ошибки обучения (fitError), 
максимального числа итераций обучения (maxIter), 
размера популяции (PopSize) и числа поколений (GNum); 
производится оптимизация гиперпараметров импортированной нейронной сети и вывод сопутствующей информации.

В ПО используется API Neuroph; Разработчики API Neuroph предоставляют редактор нейронных сетей 
Neuroph Studio ( http://neuroph.sourceforge.net/download.html ), 
в котором возможен просмотр входных и выходных файлов в формате "*.nnet". 
Формат файлов *.dset, представляющий файл выборки, был разработан в процессе создания программной реализации 
алгоритма оптимизации. Файлы этого формата можно создавать и открывать с помощью стандартного текстового редактора. 