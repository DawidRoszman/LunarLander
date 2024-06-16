# LunarLander – Algorytmy sterujące lądowaniem

## Opis zadania

W tym przypadku warto skorzystać z paczki Gym/ Gymnasium:

https://gymnasium.farama.org/environments/box2d/lunar_lander/

https://www.gymlibrary.dev/environments/box2d/lunar_lander/

Chcemy pomyślnie wylądować sondą na księżycu. Napisz program, który to wykona. Do
rozpatrzenia są trzy możliwości:

- własny skrypt,
- algorytmy reinforcement learning (w tym także z siecią neuronową)
- kontroler rozmyty (wymaga napisania bazy i reguł)
- strategie metaheurystyczne (PSO lub Algorytm Genetyczny)

Na końcu zademonstruj działanie algorytmów na animacjach (możesz zapisać jaki video lub
gify)

## Pierwsze podejście

W moim pierwszym podejściu spróbowałem samemu napisać skrypt, który na podstawie nagrody otrzymanej
za ruch zwiększa lub zmniejsza prawdopodobieństwo wykonania tego ruchu przy kolejnych krokach.
To podejście miało jednak wady. Jedną z wad było zmienianie prawdopodobieństw wyłącznie na podstawie
ostatniej nagrody, bez brania pod uwagę skutków długo trwałych.

W pierwszej próbie stworzyłem skrypt, który uczy się przez wzmacnianie pozytywnych akcji.
Niestety, metoda ta działała tylko na podstawie ostatniej nagrody
i nie uwzględniała długoterminowych skutków.

Poprawiłem to stosując strategię Epsilon-Greedy. Uczy ona zarówno z wyborów
przynoszących natychmiastową nagrodę, jak i z eksploracji nowych.
Dodatkowo, wprowadzono mechanizm cofania się do udanych wyborów z
poprzedniej iteracji, jeśli aktualna próba jest mniej efektywna.
Wraz z czasem agent mniej eksploruje, a bardziej polega na
zdobytej wiedzy (wyższe prawdopodobieństwo udanych akcji).

## Reinforcement Learning

Przy drugim podejściu postanowiłem użyć biblioteki `stable_baselines3`, która umożliwia mi trenowanie
agenta używając algorytmów takich jak PPO lub A2C. Najpierw wytrenowałem model na algorytmie `A2C`.
Przy tym modelu rakiecie udawało się lądować, niestety było to wolne oraz nie zawsze kończyło się sukcesem.
Kolejnym krokiem było wytrenowanie modelu na algorytmie `PPO`. Model ten uczył się szybciej oraz z większą
precyzją.

## Kolejne kroki

Po udanym wytrenowaniu modelu na algorytmie `PPO` postanowiłem zmienić parametry środowiska, aby sprawdzić
czy model jest w stanie nauczyć się lądować w innych warunkach. Zmieniłem m.in. grawitację, oraz dodałem wiatr. Obecnemu modelowi czasami udaje się lądować, jednak nie jest to zbyt skuteczne. Postanowiłem go dotrenować na nowych warunkach. Trenowałem ten sam model zmieniająć parametry środowiska. Po całkiem długim uczeniu model zaczął lądować w nowych warunkach. Obecnie model jest w stanie lądować w różnych warunkach (nawet ekstremalnych).

## Materiały

1. https://www.youtube.com/watch?v=nRHjymV2PX8
2. https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/
3. https://gymnasium.farama.org/environments/box2d/lunar_lander/
4. https://github.com/sudharsan13296/Hands-On-Reinforcement-Learning-With-Python/blob/master/11.%20Policy%20Gradients%20and%20Optimization/11.2%20Lunar%20Lander%20Using%20Policy%20Gradients.ipynb
5. https://stable-baselines.readthedocs.io/en/master/
6. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
7. https://codecrucks.com/mamdani-fuzzy-inference-method-example/
