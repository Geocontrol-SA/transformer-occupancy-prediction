# Transformer para Predição de Nível de Ocupação em Pontos de Ônibus - Abordagens e Decisões

## Introdução

Este documento apresenta as abordagens e decisões tomadas durante a implementação de um modelo Transformer para predição de nível de ocupação em pontos de ônibus, desenvolvido a partir do zero, sem a utilização de fine-tuning de modelos pré-treinados. O objetivo inicial foi abranger toda a complexidade inerente ao problema, buscando capturar os múltiplos aspectos relacionados à predição de ocupação em diferentes contextos.

## Abordagem Inicial

A primeira abordagem envolveu o desenvolvimento completo de um modelo Transformer, com a proposta de treinar o modelo diretamente a partir dos dados coletados, sem recorrer a modelos pré-existentes. A arquitetura revelou-se particularmente complexa, exigindo camadas adicionais para lidar com dados categóricos de alta cardinalidade, além de outras estratégias computacionais que precisaram ser implementadas para garantir a viabilidade do processamento.

A massa de dados utilizada para o treinamento consistia em todas as viagens do sistema ao longo de um ano, o que resultou em um volume considerável de informações a serem processadas. No entanto, rapidamente constatou-se a inviabilidade da abordagem devido às limitações de hardware e ao extenso tempo necessário para completar o treinamento. A combinação de alta complexidade arquitetural com um conjunto de dados massivo tornou inviável realizar qualquer avaliação de desempenho, mesmo após várias épocas de treinamento.

## Abordagem Simplificada

Diante da impossibilidade de prosseguir com a abordagem inicial, optou-se por simplificar o modelo, reduzindo o número de camadas e a complexidade geral da arquitetura. A nova configuração foi planejada para lidar com um conjunto de dados significativamente menor, restrito às viagens de um único itinerário ao longo de um único mês. Essa decisão proporcionou uma estrutura mais manejável, permitindo a avaliação do desempenho do modelo e o ajuste de hiperparâmetros de forma eficiente.

Essa simplificação não apenas viabilizou o treinamento, mas também criou uma base metodológica sólida para futuras melhorias e expansões do modelo, permitindo que lições aprendidas na abordagem simplificada sejam aplicadas em arquiteturas mais robustas no futuro.

## Considerações Finais

O modelo inicial, dada sua inviabilidade prática, será apenas mencionado de maneira breve, uma vez que não foi possível concluir seu treinamento e avaliação. Por outro lado, o modelo simplificado, que demonstrou maior viabilidade experimental, será apresentado detalhadamente, com ênfase na estrutura arquitetural, metodologia de treinamento e resultados obtidos.

