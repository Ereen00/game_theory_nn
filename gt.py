import torch
import torch.nn as nn

# Oyun parametreleri
COOPERATE = 0
DEFECT = 1

# Ajan sınıfı
class Agent(nn.Module):
    def __init__(self, learning_rate=0.01):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(3, 5)  # 3 giriş (rakip seçimleri, kendi seçimleri), 5 gizli nöron
        self.fc2 = nn.Linear(5, 2)   # 2 çıkış (COOPERATE veya DEFECT)
        self.last_choice = COOPERATE  # Başlangıçta COOPERATE
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        x = torch.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x

    def learn(self, reward):
        # Ödül-ceza sistemini kullanarak öğrenme
        target = torch.tensor([reward, 0.0]) if reward > 0 else torch.tensor([0.0, -reward])
        target = target.unsqueeze(0)  # Boyutları eşitle
        output = self.forward(torch.tensor([self.last_choice] * 3, dtype=torch.float32).unsqueeze(0))
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset(self):
        # Ajanın ağırlıklarını sıfırla
        self.fc1.weight.data.fill_(0.0)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.weight.data.fill_(0.0)
        self.fc2.bias.data.fill_(0.0)
        self.last_choice = COOPERATE  # Başlangıçta COOPERATE

# Oyun simülasyonu
class Game:
    def __init__(self, agents):
        self.agents = agents
        self.scores = [0, 0, 0]

    def play_round(self):
        match_results = []  # Maç sonuçlarını tutacak
        for i in range(3):
            for j in range(i + 1, 3):
                inputs_i = torch.tensor([self.agents[j].last_choice, 
                                         self.agents[i].last_choice, 
                                         self.agents[i].last_choice], dtype=torch.float32)
                inputs_j = torch.tensor([self.agents[i].last_choice, 
                                         self.agents[j].last_choice, 
                                         self.agents[j].last_choice], dtype=torch.float32)

                choice_i = torch.argmax(self.agents[i](inputs_i)).item()
                choice_j = torch.argmax(self.agents[j](inputs_j)).item()

                # Seçimleri kaydet
                self.agents[i].last_choice = choice_i
                self.agents[j].last_choice = choice_j

                # Puanları hesapla
                if choice_i == COOPERATE and choice_j == COOPERATE:
                    self.scores[i] += 3
                    self.scores[j] += 3
                    result = 'C vs C: 3 - 3'
                    reward_i = 3
                    reward_j = 3
                elif choice_i == DEFECT and choice_j == DEFECT:
                    self.scores[i] += 1
                    self.scores[j] += 1
                    result = 'D vs D: 1 - 1'
                    reward_i = 1
                    reward_j = 1
                elif choice_i == COOPERATE and choice_j == DEFECT:
                    self.scores[i] += 0
                    self.scores[j] += 5
                    result = 'C vs D: 0 - 5'
                    reward_i = 0
                    reward_j = 5
                elif choice_i == DEFECT and choice_j == COOPERATE:
                    self.scores[i] += 5
                    self.scores[j] += 0
                    result = 'D vs C: 5 - 0'
                    reward_i = 5
                    reward_j = 0

                match_results.append(f"Ajan {i + 1} ve Ajan {j + 1} maç sonucu: {result}")

                # Ajanların öğrenmesi
                self.agents[i].learn(reward_i)
                self.agents[j].learn(reward_j)

        return match_results, self.scores

# Simülasyonu başlat
def main():
    agents = [Agent() for _ in range(3)]  # Ajanları oluştur
    game = Game(agents)  # Oyun nesnesini oluştur

    rounds = 10  # Oynanacak tur sayısı
    for round_number in range(1, rounds + 1):
        match_results, scores = game.play_round()
        print(f"TUR {round_number}:")
        for match in match_results:
            print(match)
        print(f"Toplam Puanlar: {scores}\n")

if __name__ == "__main__":
    main()


# Her 5 turda bir ajanların öğrenme süreçlerini sıfırlama
# if round_number % 5 == 0:
#     for agent in agents:
#         agent.reset()
#     print("Ajanlar sıfırlandı.\n")