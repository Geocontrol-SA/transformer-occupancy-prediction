import pandas as pd
import json
from datetime import datetime
import argparse


# exemplo
# human_speaks = "Sabendo que está chovendo. Qual a lotação da linha 507I para o ponto 2333243 para domingo às %hora?"
human_speaks = "Sabendo que está %clima_tempo e a temperatura é %temperatura. Qual a lotação da linha %linha para o ponto %ponto para %dia às %hora?"

#exemplo
# bot_speaks = "O ponto 2333243 da linha 507I, que corresponde ao percentual 34% do itinerario, a lotação é %lotação"
bot_speaks = "O ponto %ponto da linha %linha, que corresponde ao percentual %percentual do itinerario, a lotação é %lotação."

days_of_week = {
    0: "domingo",
    1: "segunda",
    2: "terça",
    3: "quarta",
    4: "quinta",
    5: "sexta",
    6: "sabado"
}

def convert_to_dict(
        linha: str,
        ponto: str,
        horario: datetime,
        nivel_lotacao: int,
        percentual: float,
        humidade: float,
        temperatura: float
      ):
  clima_tempo = 'Chovendo' if humidade > 97 else 'Nublado' if humidade > 90 else 'Ensolarado'
    
  day_of_week = horario.weekday()
  dia = days_of_week[day_of_week]
  hora = horario.strftime("%H:%M")

  lotacao = 'lotado' if nivel_lotacao > 1 else 'vazio' if nivel_lotacao < 1 else 'cheio'

  
  human_says = human_speaks\
  .replace("%clima_tempo", clima_tempo)\
  .replace("%temperatura", str(temperatura))\
  .replace("%linha", linha)\
  .replace("%ponto", ponto)\
  .replace("%dia", dia)\
  .replace("%hora", hora)\

  bot_says = bot_speaks\
  .replace("%ponto", ponto)\
  .replace("%linha", linha)\
  .replace("%lotação", lotacao)\
  .replace("%percentual", str(percentual))

  return {
    'instruction': human_says,
    'input': '',
    'output': bot_says,        
  }

  # return {
  #     "human": human_says,
  #     "bot": bot_says
  # }


    
    # extract day of week 
    
    # human_says = human_speaks % (linha, ponto, dia, hora)
    # bot_says = bot_speaks % (lotação)
    # return {
    #     "human": human_speaks,
    #     "bot": bot_speaks
    # }

def convert_to_text(
        linha: str,
        ponto: str,
        horario: datetime,
        nivel_lotacao: int,
        percentual: float,
        humidade: float,
        temperatura: float
      ):
    speaks = convert_to_dict(linha, ponto, horario, nivel_lotacao, percentual, humidade, temperatura)
    
    return f"""<human>: {speaks['human']}\n<bot>:{speaks['bot']}"""



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--file', type=str, help='Nome arquivo de saida')
    args = parse.parse_args()
    print(f"args: {args}")


    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('data/vehicle_bus_stop_occupation_20231101_20231201.csv')
    print(df.columns)
    print(df.dtypes)

    print(df.head())
    print(df['temperature'][10])
    with open(args.file, 'w') as f:
      conversations = []  
      for i, row in df.iterrows():
        linha = row['itinerary_code']
        ponto = row['busstop_code']
        horario = datetime.fromisoformat(row['reading_time'])
        nivel_lotacao = row['occupation']
        percentual = 100 * row['normalized_location']
        humidade = row['humidity']
        temperatura = row['temperature']
        # conversations.append(convert_to_dict(linha, ponto, horario, nivel_lotacao, percentual, humidade, temperatura))
        conversation = convert_to_dict(linha, ponto, horario, nivel_lotacao, percentual, humidade, temperatura)
        json.dump(conversation, f)
        f.write('\n')
        f.flush()

      
      
      # f.write(json.dumps(conversations).encode('utf-8').decode('utf-8'))
      # f.write('\n')
      # f.flush()




       
    

