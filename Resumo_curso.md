# Resumo do curso CEIA/Meta — AI Glasses Brasil 2026 (Unidades 2 e 5–13)
### Mapeamento de conteúdo para o projeto HigenAI

**Contexto do projeto:** HigenAI usa os óculos Meta para observar (câmera) o gesto de higienização das mãos de profissionais de saúde, verificar se os passos da técnica da OMS foram seguidos, responder por áudio, e manter um agente de memória que registra higienizações e emite lembretes.

**Como ler este documento:** cada item traz (a) o conceito, de forma concisa, e (b) sua relação com um projeto como o HigenAI. Cada item é sinalizado como:
- 🟢 **Aplicável diretamente** — toca em algo que o HigenAI provavelmente vai usar ou decidir.
- ⚪ **Fundamento teórico geral** — contexto/vocabulário útil, mas não necessariamente parte da implementação.

Este é um mapeamento de conteúdo, não uma recomendação de arquitetura — as escolhas de implementação ficam para a equipe.

---

## 1. Arquitetura de agentes

Base conceitual de "o que é um agente" e "como ele é estruturado por dentro" (Unidades VI e VII).

### 1.1 Definição e propriedades de agentes inteligentes
- **Conceito:** um agente é uma entidade que percebe o ambiente via sensores e age via atuadores, de forma autônoma, num ciclo percepção → decisão → ação, buscando cumprir um objetivo. Propriedades-chave: autonomia, reatividade, proatividade, capacidade de aprendizagem e habilidade social.
- **Relação com o HigenAI:** ⚪ Fundamento teórico geral, mas mapeia direto no vocabulário do projeto — a câmera dos óculos é o "sensor", os alto-falantes são o "atuador", e o ciclo percepção-decisão-ação é literalmente câmera → verificação do gesto → resposta em áudio.

### 1.2 Ambientes de agentes (dimensões de classificação)
- **Conceito:** ambientes se classificam por observabilidade (total/parcial), determinismo (determinístico/estocástico), episodicidade (episódico/sequencial), dinamicidade (estático/dinâmico), continuidade (discreto/contínuo) e número de agentes (único/multiagente).
- **Relação com o HigenAI:** ⚪ Fundamento teórico. Mas é um exercício útil: o "ambiente" do HigenAI é **parcialmente observável** (câmera só vê o que está no campo de visão), **sequencial** (os passos da técnica da OMS dependem de ordem/histórico), **dinâmico** (mãos se movem continuamente) — isso indica que um agente puramente reativo (sem memória de estado) dificilmente basta para verificar uma sequência de passos.

### 1.3 Tipos de agentes (reflexo simples → aprendizagem)
- **Conceito:** cinco categorias cumulativas — reflexo simples (regra condição-ação, sem memória), reflexo baseado em modelo (mantém estado interno), baseado em objetivos (planeja para atingir metas), baseado em utilidade (compara diferentes formas de atingir a meta) e de aprendizagem (melhora com experiência via elemento crítico + elemento de aprendizado).
- **Relação com o HigenAI:** 🟢 Aplicável diretamente como vocabulário de design: verificar uma sequência de passos da técnica da OMS exige, no mínimo, um **agente baseado em modelo** (lembrar quais passos já foram observados nesta higienização). O componente de "agente de memória" que registra higienizações e emite lembretes se encaixa na ideia de manter estado/memória entre sessões.

### 1.4 Sistemas multiagentes (MAS)
- **Conceito:** múltiplos agentes autônomos coexistindo, cooperando ou competindo; benefícios (paralelismo, robustez, especialização) e desafios (coordenação, quebra de contexto, custo, depuração, alinhamento de objetivos).
- **Relação com o HigenAI:** ⚪ Fundamento teórico geral — o escopo do HigenAI (visão + verificação + agente de memória) pode ser resolvido com poucos componentes especializados, mas o conceito de dividir responsabilidades (ex.: um componente que reconhece o gesto, outro que decide a resposta em voz, outro que gerencia lembretes) é relevante caso a equipe opte por modelar isso como múltiplos agentes cooperando.

### 1.5 Arquiteturas de agentes: reflexiva, baseada em modelo, híbrida
- **Conceito:** arquitetura **reflexiva** decide só com a percepção atual (rápida, sem memória); arquitetura **baseada em modelo** mantém estado interno e permite planejamento simples; arquitetura **híbrida** combina uma camada reativa (resposta imediata a estímulos urgentes) com uma camada deliberativa (planejamento) e uma camada de coordenação entre as duas. BDI (Belief-Desire-Intention) é citado como exemplo de arquitetura híbrida/deliberativa.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente ao desenho do "verificador de gestos": ele precisa tanto reagir rápido a cada frame/gesto (camada reativa) quanto manter noção de "quais passos da OMS já foram cumpridos nesta sessão de higienização" (camada com estado/modelo) — um caso de uso quase didático para arquitetura híbrida, ainda que a escolha final da equipe possa ser mais simples.

### 1.6 Critérios para escolher a arquitetura
- **Conceito:** a escolha depende de complexidade/dinamismo do ambiente, disponibilidade de dados/conhecimento prévio, requisitos de latência e custos de desenvolvimento/computação.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente como checklist de decisão: a resposta por voz do HigenAI tem restrição de latência perceptível pelo usuário (ninguém quer esperar segundos para saber se lavou a mão certo), o que pesa a favor de componentes reativos/rápidos para o feedback imediato, com uma camada mais "pensante" rodando em paralelo para lembretes e registro.

### 1.7 Padrões de projeto para agentes
- **Conceito:** três padrões recorrentes — **Planejador-Executor** (separar "decidir o plano" de "executar passo a passo"), **Supervisor-Trabalhador** (um agente central delega subtarefas a agentes especializados) e **Agente com Ferramentas/Skills** (o agente central chama módulos/APIs externas quando precisa de uma capacidade que não tem nativamente).
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — o padrão "Agente com Ferramentas/Skills" é o mais próximo do que o HigenAI provavelmente precisa: um agente central que, quando necessário, aciona a "ferramenta" de visão computacional (para checar o gesto), a "ferramenta" de síntese de voz (para responder) e a "ferramenta" de registro/memória (para lembretes). Não é uma recomendação de que a equipe deva usar exatamente esse padrão — é o mapeamento de que o conteúdo do curso descreve esse padrão como uma opção testada.

---

## 2. Ferramentas de agentes

Ecossistema de bibliotecas e frameworks para construir, orquestrar e dar memória a agentes baseados em LLM (Unidades V, VIII, IX, X e XI). **Importante:** este conteúdo é majoritariamente sobre o ecossistema de LLMs em nuvem/servidor (Python, APIs como OpenAI) — o HigenAI roda on-device nos óculos/celular, então grande parte disso é fundamento teórico geral que ajuda a entender o vocabulário do campo, não peças que necessariamente entram no app Android final.

### 2.1 Ferramentas de manipulação de modelos de linguagem
- **Conceito:** Hugging Face Transformers (biblioteca para carregar/treinar/usar LLMs), OpenAI API, Google Cloud Natural Language API, IBM Watson NLU — cada uma com características e aplicações próprias (geração de texto, análise de sentimento, extração de entidades).
- **Relação com o HigenAI:** ⚪ Fundamento teórico geral. Essas ferramentas pressupõem chamadas a serviços em nuvem ou execução em Python/servidor — não é o caminho natural para um app Android on-device, mas explica o vocabulário (pipeline, tokenização, fine-tuning) usado depois no material de IA on-device.

### 2.2 Fine-tuning, personalização e engenharia de prompt (para LLMs de texto)
- **Conceito:** ajuste fino (retreinar com dados de domínio), personalização (adaptar estilo/tom), adaptação por engenharia de prompt (sem alterar o modelo) e ajuste por dados de aprendizado ativo (feedback do usuário).
- **Relação com o HigenAI:** ⚪ Fundamento teórico geral — mais relevante se o agente de memória/lembretes do HigenAI usar algum LLM para gerar mensagens ou interpretar comandos de voz; o conceito de "engenharia de prompt" é o mais diretamente transferível, caso a solução inclua algum modelo de linguagem para compor as respostas faladas.

### 2.3 Ecossistema de orquestração: LangFlow, LangChain, CrewAI, LangGraph, MANGABA AI
- **Conceito:** LangFlow (baixo código, visual), LangChain (framework Python com prompts/chains/tools/agentes), CrewAI (orquestra "equipes" de agentes com memória compartilhada), LangGraph (fluxos como grafos de estados, com nós, arestas, loops e condições), MANGABA AI (framework brasileiro para times de agentes com workflows automáticos).
- **Relação com o HigenAI:** ⚪ Fundamento teórico geral. São ferramentas pensadas para orquestrar chamadas a LLMs via API — úteis para prototipar a lógica do agente de memória/lembretes em um notebook/servidor durante o desenvolvimento (ex.: simular o fluxo de decisão antes de portar para Kotlin), mas não rodam nativamente nos óculos/Android. O conceito de LangGraph (fluxo como grafo de estados, com nós e transições condicionais) é conceitualmente parecido com o "ciclo de vida" do DAT (seção 3) e pode servir de inspiração para modelar os estados da verificação de higienização.

### 2.4 Bases de dados vetoriais e embeddings
- **Conceito:** embeddings representam texto/imagem como vetores onde proximidade = similaridade semântica; métricas de comparação (distância euclidiana, similaridade de cosseno, produto interno); bancos vetoriais (ChromaDB, Qdrant, pgvector, FAISS, Weaviate, Milvus, Pinecone) para indexar e buscar por similaridade.
- **Relação com o HigenAI:** ⚪ Fundamento teórico geral, com uma aplicação possível: se o agente de memória do HigenAI precisar responder perguntas como "quantas vezes o profissional higienizou as mãos hoje" ou comparar padrões de comportamento ao longo do tempo, bases vetoriais/embeddings são uma técnica candidata para dar "memória semântica" ao agente — mas para um registro estruturado (timestamps, contagens), um banco relacional simples também resolveria; a escolha fica para a equipe.

### 2.5 RAG (Retrieval-Augmented Generation) e boas práticas de relevância
- **Conceito:** técnica de buscar trechos relevantes numa base externa antes de gerar uma resposta, em vez de depender só do conhecimento "fixo" do modelo; boas práticas incluem chunking (segmentar documentos), uso de metadados/filtros e re-ranking (reordenar resultados por relevância real).
- **Relação com o HigenAI:** ⚪ Fundamento teórico geral. Mais relevante se a equipe quiser que o assistente responda perguntas sobre o protocolo da OMS consultando um documento de referência (ex.: "por que esse passo é importante"), em vez de responder só via confirmação/feedback do gesto.

### 2.6 Tipos de memória de agentes
- **Conceito:** memória de curta duração (janela de contexto, volátil), memória de longa duração/persistente (vetorial ou não, sobrevive entre sessões), memória episódica (eventos específicos com contexto de tempo — "o que aconteceu quando"), memória semântica (fatos gerais e estáveis — "o que é X").
- **Relação com o HigenAI:** 🟢 Aplicável diretamente como vocabulário de design para o "agente de memória" citado na descrição do projeto: registrar cada higienização com timestamp é essencialmente **memória episódica**; saber que "o profissional João geralmente higieniza corretamente" seria mais próximo de memória semântica; e o estado da higienização em andamento (quais passos já foram vistos) é memória de curto prazo.

### 2.7 Estratégias de RAG, gestão de janela de contexto e persistência/retenção
- **Conceito:** estratégias de recuperação (busca por similaridade, hierárquica, combinação de memória episódica+semântica), gestão de janela de contexto (truncamento, sumarização, segmentação por tópico), e práticas de persistência/versionamento/retenção de dados ao longo do tempo.
- **Relação com o HigenAI:** ⚪ Fundamento teórico geral — relevante principalmente se o agente de lembretes usar algum modelo de linguagem com histórico de conversa; para um registro de eventos de higienização, o conceito mais transferível é "por quanto tempo reter os dados" e "como versionar correções" (ex.: se um registro de higienização for corrigido/invalidado depois).

### 2.8 Qualidade de dados, mitigação de viés, governança e LGPD
- **Conceito:** processos para manter a memória do agente correta e atualizada, mitigar vieses herdados dos dados/modelos, e um checklist de conformidade com a LGPD: minimização de dados, consentimento e transparência, anonimização/pseudonimização, segurança dos dados armazenados, direitos dos titulares, privacy by design, documentação/DPIA.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — o projeto capta **imagens de profissionais de saúde** e registra **dados de comportamento individual** (quem higienizou, quando, se corretamente). Isso é dado pessoal (e potencialmente sensível, dependendo de como for tratado) sob a LGPD. O checklist da seção 11.5 do material (minimização — ex. registrar só o necessário, talvez um ID em vez do nome; consentimento explícito dos profissionais; anonimização quando possível; segurança dos dados de vídeo/registro) é diretamente relevante para qualquer decisão de como o agente de memória armazena e usa essas informações.

---

## 3. Toolkit dos óculos (Meta Wearables Device Access Toolkit — DAT)

Conteúdo específico de hardware e SDK dos óculos Ray-Ban Meta usados no programa (Unidade XIII). Praticamente todo este bloco é 🟢 **aplicável diretamente**, pois é a camada de acesso ao hardware que o HigenAI precisa para captar vídeo e tocar áudio.

### 3.1 Hardware dos óculos e suas limitações
- **Conceito:** Ray-Ban Meta tem câmera ultra-wide de 12 MP, array de 5 microfones e alto-falantes open-ear — **sem display**. Três limitações moldam qualquer solução: saída só por áudio, dependência de pareamento Bluetooth com um app companion no celular (os óculos não executam código próprio), e bateria/banda limitadas.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — define exatamente a forma da solução do HigenAI: entrada por câmera (observar o gesto), saída só por voz (confirmar/orientar o profissional), sem qualquer elemento visual. Reforça que toda resposta do sistema — incluindo lembretes — precisa ser pensada como áudio curto e objetivo.

### 3.2 O que o DAT permite e o que não permite
- **Conceito:** o DAT (em developer preview) permite streaming de vídeo da câmera, captura de foto, registro do app e gestão de sessão. **Não** permite rodar código nos óculos, nem acessar "Hey Meta"/Meta AI, nem mais de uma sessão simultânea. Microfone e alto-falantes **não** passam pelo DAT — usam os perfis Bluetooth padrão do Android.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — implica que o HigenAI terá dois caminhos técnicos distintos e não intercambiáveis: vídeo via API do DAT (Stream/Session) e áudio via APIs padrão de Bluetooth do Android (AudioManager/AudioRecord/AudioTrack, perfil HFP). Qualquer wake word própria ("Hey HigenAI", por exemplo) precisaria ser implementada pela equipe, já que o "Hey Meta" não é acessível.

### 3.3 Arquitetura da solução: óculos ⇄ Bluetooth ⇄ app companion
- **Conceito:** os óculos são só um periférico de I/O (capturam e reproduzem); toda a computação — modelos de IA e lógica do agente — roda no app Android companion no celular, conectado via Bluetooth.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — define onde a "inteligência" do HigenAI (reconhecimento do gesto, verificação dos passos da OMS, agente de memória) precisa rodar: no celular, não nos óculos. Isso também implica orçamento de latência de ponta a ponta (captura → Bluetooth → inferência → TTS → Bluetooth → áudio) como uma restrição real de design.

### 3.4 Setup do ambiente e distribuição do SDK
- **Conceito:** o DAT é distribuído via GitHub Packages (exige autenticação mesmo para pacotes públicos, com um personal access token), configurado no Gradle (Kotlin DSL) com artefatos mwdat-core, mwdat-camera e mwdat-mockdevice, e o manifest precisa de permissões + metadados de atestação (APPLICATION_ID/CLIENT_TOKEN, dispensáveis em Developer Mode) e um intent filter com URI scheme próprio.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — é o passo mecânico necessário antes de qualquer integração real com os óculos; relevante para quem for configurar o projeto Android do HigenAI.

### 3.5 Registro e ciclo de sessão
- **Conceito:** o SDK é inicializado uma vez por processo (`Wearables.initialize`); o registro do app acontece dentro do app Meta AI; o acesso aos sensores ocorre dentro de uma `DeviceSession`, com ciclo de estados IDLE → STARTING → STARTED ⇄ PAUSED → STOPPING → STOPPED (terminal). Erros e mudanças de estado chegam por Flows separados, e o app deve reagir ao estado observado (não presumir a causa).
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — é a máquina de estados que qualquer app Android construído sobre o DAT precisa respeitar; relevante em particular para lidar com interrupções (ex.: profissional de saúde tira os óculos no meio de uma higienização) sem travar a experiência.

### 3.6 Câmera: Stream, resolução, frame rate e captura de foto
- **Conceito:** vídeo chega como um `Stream` anexado à sessão, com resolução configurável (HIGH 720×1280 / MEDIUM 504×896 / LOW 360×640) e frame rate entre 2–30 fps; a banda do Bluetooth Classic é o fator limitante, com rebaixamento automático de qualidade quando falta banda.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — a verificação do gesto de higienização depende diretamente de qual resolução/fps escolher; o material observa que pipelines de visão computacional raramente precisam de mais que ~15 fps em MEDIUM/LOW, e que pedir menos banda pode até melhorar a qualidade percebida (menos compressão por frame) — informação diretamente relevante para configurar o stream de câmera do HigenAI.

### 3.7 Áudio: perfis A2DP e HFP via Bluetooth padrão do Android
- **Conceito:** microfone e alto-falantes dos óculos não passam pelo DAT — são acessados como um headset Bluetooth comum, via A2DP (só saída, alta qualidade — mídia) ou HFP (bidirecional, voz, mas 8 kHz mono, com beamforming que isola a voz do usuário e atenua ruído ambiente).
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — se o HigenAI usar comando de voz do profissional (além da resposta falada do sistema), o caminho é HFP, com a limitação de qualidade de "ligação telefônica" — suficiente para fala, mas não para captar sons ambientes. O material também alerta que, ao combinar vídeo (câmera) e voz, é preciso configurar o HFP **antes** de iniciar o streaming de vídeo, para evitar comportamento intermitente — um detalhe de implementação relevante se o HigenAI usar os dois canais ao mesmo tempo (o que é o caso: câmera para observar o gesto, áudio para responder).

### 3.8 Mock Device Kit (desenvolvimento sem hardware)
- **Conceito:** simula a pilha inteira do DAT (streaming de câmera, captura de foto, permissões, estado do dispositivo, gestos de toque) sem óculos físicos — usando vídeo H.265, a câmera do próprio celular, ou imagens fixas. Não simula áudio (porque áudio nunca passou pelo DAT).
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — permite à equipe começar a desenvolver e testar toda a lógica de verificação de gestos (usando um vídeo de exemplo de alguém higienizando as mãos, ou a própria câmera do celular) antes de ter acesso aos óculos físicos, o que é exatamente a situação da fase de preparação para o hackathon.

---

## 4. IA on-device

Como rodar modelos de visão, voz e IA em geral no celular (não na nuvem), incluindo os fundamentos de Android necessários para sustentar isso (Unidade XII).

### 4.1 Fundamentos de Android relevantes ao pipeline de IA
- **Conceito:** coroutines/`suspend`/`Flow` para lidar com fluxos assíncronos contínuos sem travar a UI; `ViewModel`+`StateFlow` para estado que sobrevive a mudanças de tela; ciclo de vida da Activity; permissões em runtime (câmera, microfone, Bluetooth) com tratamento de negação simples vs. permanente.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — é a infraestrutura básica sobre a qual qualquer parte do app HigenAI (câmera, áudio, inferência) vai ser construída em Kotlin/Android; `Flow` em particular é o mecanismo natural para tratar o stream contínuo de frames da câmera e blocos de áudio.

### 4.2 Background e eficiência de energia
- **Conceito:** Android restringe fortemente execução em segundo plano (Doze, App Standby); os dois mecanismos sancionados são foreground service (trabalho contínuo, com notificação) e WorkManager (trabalho adiável). Inferência "em rajada" (bursty) — picos curtos + ociosidade — é mais eficiente em bateria e evita throttling térmico do que inferência contínua.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — um app que monitora higienização de mãos ao longo de um turno de trabalho provavelmente precisa rodar em segundo plano (foreground service, já que é contínuo e perceptível ao usuário) e deve evitar rodar o modelo de visão em todo frame o tempo todo — o padrão "gatilho, não loop" (ex.: só rodar a verificação quando detectar movimento perto de uma pia) é diretamente relevante para bateria e para não superaquecer o celular durante um turno inteiro.

### 4.3 Edge AI: por que e como reduzir modelos
- **Conceito:** rodar IA no próprio dispositivo troca a "nuvem" (mais poder, mas com latência de rede) por "borda" (resposta imediata, privacidade, funciona offline, sem custo de API — mas limitado por memória, calor e energia). Técnicas de redução: quantização (FP32→INT8→INT4, reduzindo tamanho e ganhando velocidade, com trade-off de qualidade), destilação (modelo pequeno treinado para imitar um grande), pruning (remover pesos pouco relevantes) e LoRA/adaptadores (especializar sem re-treinar o modelo base inteiro).
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — é a base teórica e prática para levar qualquer modelo de reconhecimento de gesto ao celular: memória e latência limitadas do celular pareado aos óculos tornam quantização (pelo menos INT8) praticamente obrigatória, e o material recomenda começar sempre pela opção mais barata (PTQ INT8) antes de partir para técnicas mais caras.

### 4.4 Formatos de modelo e hardware de execução
- **Conceito:** o formato do arquivo do modelo determina o runtime e o hardware onde ele roda bem: `.tflite`/`.litertlm` (LiteRT/Google) para celular, GGUF (llama.cpp) para LLMs em notebook, ONNX para portabilidade entre ambientes. Hardware de execução: CPU (universal, fallback), GPU (paraleliza modelos maiores via delegate), NPU (mais eficiente energeticamente no celular, mas com acesso em transição no Android).
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — como o modelo de visão do HigenAI precisa rodar no celular pareado aos óculos, o caminho natural indicado pelo material é `.tflite` via LiteRT, com CPU como alvo garantido e GPU como possível bônus (a decisão de qual usar exige medir no aparelho real, algo que o próprio material recomenda repetidamente).

### 4.5 Visão computacional e voz on-device
- Estes dois assuntos ganharam módulos próprios e bem mais extensos no material (Unidade XII, tópicos 12.12 e 12.13) do que os outros itens desta seção — por isso foram destacados como a **Seção 5** abaixo, em vez de ficarem resumidos em uma linha aqui.

---

## 5. Processamento de imagem (visão computacional) e processamento de voz (STT/TTS) on-device

Esta seção é, no material, praticamente um "catálogo de técnicas" dedicado — dois módulos completos (Unidade XII, tópicos 12.12 e 12.13) com objetivos de aprendizagem próprios, tabelas comparativas, exemplos de código e um "na prática" de ponta a ponta. Por isso ficou destacada separadamente das demais partes de IA on-device (Seção 4): é provavelmente o conteúdo com **maior densidade de aplicação direta** ao HigenAI, já que o projeto é literalmente "câmera nos óculos → reconhecimento de gesto → resposta em voz".

### 5.1 Catálogo de tarefas de visão computacional
- **Conceito:** sete tarefas, cada uma com o que resolve e o peso computacional esperado on-device: **classificação** (rótulo da imagem inteira; leve — ex.: MobileNet), **detecção de objetos** (caixa + classe + confiança; leve a médio — ex.: YOLO), **segmentação** (máscara pixel a pixel; médio a pesado), **OCR** (texto localizado e transcrito; médio — ex.: ML Kit), **landmarks** (pontos-chave de rosto/mãos/pose — 468 pontos da malha facial, juntas dos dedos, articulações; leve a médio — via pipelines do MediaPipe), **captioning** (frase descrevendo a cena; pesado, exige modelo multimodal, latência de segundos) e **rastreamento/tracking** (mantém identidade de um objeto entre frames; custo adicional moderado sobre a detecção).
- **Relação com o HigenAI:** 🟢 Aplicável diretamente. O material é explícito: "com a câmera em primeira pessoa, as mãos do próprio usuário aparecem no campo de visão — dá para reconhecer gestos das mãos como comandos" (citando exatamente esse cenário de gestos das mãos). Verificar os passos da técnica de higienização da OMS mapeia mais naturalmente para **landmarks de mãos** (pontos-chave para inferir gestos específicos — esfregar palma-palma, dorso, entre dedos, polegar, unhas) do que para detecção de objetos genérica. Uma alternativa citada no material é tratar cada passo como uma **classe de um classificador/detector fine-tunado** (ex.: YOLO treinado com vídeos dos próprios passos da OMS). O material não recomenda uma opção sobre a outra — apenas descreve as duas como caminhos existentes; a escolha é da equipe.

### 5.2 YOLO, transfer learning e fine-tuning
- **Conceito:** YOLO ("You Only Look Once") é a família de referência para detecção em tempo real (uma única passada da rede produz todas as caixas e classes), com variantes de tamanho n/s/m/l/x trocando precisão por velocidade (nano = candidata natural para CPU de celular). Transfer learning é aproveitar um modelo pré-treinado em dataset genérico (ex.: COCO) como ponto de partida; fine-tuning é re-treiná-lo com um dataset próprio e específico — segundo o material, "algumas centenas de imagens anotadas e algumas dezenas de épocas" já bastam para um protótipo funcional, viável no prazo de um hackathon (inclusive com GPU gratuita do Google Colab).
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — se a equipe optar pelo caminho de classificação/detecção (em vez de landmarks) para reconhecer os passos da OMS, este é o processo descrito: gravar/anotar um dataset próprio dos gestos da técnica (em primeira pessoa, várias distâncias/iluminações — o material enfatiza que "dataset parecido com a realidade vale mais que dataset grande") e aplicar fine-tuning sobre uma YOLO nano pré-treinada.

### 5.3 Três caminhos de implementação: ML Kit, MediaPipe, LiteRT
- **Conceito:** **ML Kit** (APIs prontas do Google — OCR, detecção genérica, landmarks de rosto/pose, image labeling; não treina nem converte nada); **MediaPipe Tasks** (pipelines prontos de visão — detecção, segmentação, landmarks de mãos/rosto/pose — com pré/pós-processamento embutido, permitindo trocar o modelo quando suportado); **LiteRT** (runtime de inferência do Google AI Edge, ex-TensorFlow Lite — roda qualquer modelo próprio, ex. YOLO fine-tunado, com máxima flexibilidade e máximo trabalho de integração).
- **Relação com o HigenAI:** 🟢 Aplicável diretamente como mapa de opções: o material indica MediaPipe como o caminho pronto para landmarks de mãos em tempo real; LiteRT seria necessário apenas se a equipe optar por um modelo de gesto próprio (fine-tunado) em vez de um pipeline pronto. Novamente, o material apresenta os três caminhos como opções — "se existe API pronta... comece por ela" é uma heurística geral do curso, não uma recomendação específica para o HigenAI.

### 5.4 Hardware de execução e o pipeline frame → resultado
- **Conceito:** CPU (universal, sempre funciona), GPU (via delegate, acelera modelos maiores em aparelhos compatíveis, mas tem custo de inicialização e nem toda operação é suportada), NPU/DSP (aceleradores dedicados, acesso ainda em transição no Android). Pipeline universal de visão: frame → pré-processamento (redimensionar/normalizar para o formato esperado pelo modelo) → inferência → pós-processamento (threshold de confiança + NMS para remover caixas duplicadas) → resultado → ação. Como a câmera gera frames mais rápido que o modelo processa, a estratégia é processar um frame por vez e descartar os demais (conflate, via `Flow`), não enfileirar tudo.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — é o desenho de baixo nível de qualquer verificador de gesto contínuo: os frames chegam via streaming Bluetooth dos óculos (Seção 3.6), passam por esse pipeline, e o resultado final se torna a fala de confirmação/correção. O material alerta especificamente que a latência de ponta a ponta (Bluetooth + inferência + TTS) precisa ser orçada em conjunto, e não medida só pela inferência isolada — relevante para o requisito de resposta "em tempo real" do HigenAI.

### 5.5 STT — transformar fala em texto
- **Conceito:** pipeline clássico (captura PCM → extração de características/mel-spectrogram → modelo acústico + decodificador → texto) vs. modelos end-to-end (uma rede mapeia áudio→texto direto, via CTC/atenção/RNN-T). Comparativo de modelos: **Whisper** (multilíngue, robusto a ruído, variantes tiny→large, mas nasceu para processamento em lote, não streaming), **Wav2Vec 2.0** (mais usado como base de pesquisa/fine-tuning do que solução mobile pronta), **Conformer/RNN-T** (projetado para streaming com baixa latência — base de ASR comercial on-device), **Vosk/Kaldi** (leve, offline, streaming nativo, qualidade inferior ao Whisper em fala espontânea, mas modelos pt-BR compactos).
- **Relação com o HigenAI:** ⚪/🟢 depende do desenho do projeto — se o HigenAI for **unidirecional** (só o sistema fala, o profissional não precisa comandar por voz), STT pode não ser necessário. 🟢 Torna-se diretamente aplicável se a equipe quiser que o profissional confirme algo por voz, peça um resumo ao agente de memória, ou corrija um registro falando — nesse caso, o material aponta Whisper tiny/base quantizado (para comandos curtos, em lote) ou Vosk (para streaming leve) como os caminhos on-device viáveis.

### 5.6 TTS — transformar texto em fala
- **Conceito:** pipeline clássico (normalização de texto → grafema-para-fonema/prosódia → modelo acústico gera mel-spectrogram → vocoder converte em forma de onda) vs. end-to-end (VITS: uma rede só, texto→onda). Comparativo: **Tacotron 2 + vocoder** (histórico, pesado para celular na forma original), **FastSpeech 2** (não-autorregressivo, rápido e estável, mas precisa de vocoder separado), **VITS** (end-to-end, base do caminho mais prático), **Piper** (baseado em VITS, otimizado para edge — nasceu para Raspberry Pi, ONNX, dezenas de idiomas incluindo pt-BR) — apontado no material como "a opção de referência para TTS neural local".
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — o material chama TTS de "o único canal rico de resposta ao usuário" em óculos sem display, exatamente o papel que a resposta por áudio do HigenAI cumpre (confirmar/corrigir a higienização). O caminho on-device citado é Piper via runtime `sherpa-onnx` (com API Kotlin), como alternativa à API nativa `TextToSpeech` do Android (mais simples de integrar, mas que pode depender de rede conforme a voz instalada no aparelho).

### 5.7 Técnicas complementares de processamento de voz
- **Conceito:** **VAD** (Voice Activity Detection — detecta se há fala ou silêncio, evitando rodar STT à toa; ex.: Silero VAD, WebRTC VAD), **diarização** (identifica "quem falou quando" em áudio com múltiplos locutores — o material nota que é "geralmente dispensável" para um assistente de um único usuário), **wake word** (frase de ativação detectada por um modelo minúsculo sempre ligado, que "acorda" o pipeline completo — o material avisa explicitamente que **"Hey Meta" pertence ao assistente da Meta e não é acessível pelo toolkit de desenvolvimento**; uma wake word própria precisaria ser construída pela equipe). Estratégias: streaming vs. lote, push-to-talk vs. sempre-escutando (mãos-livres, mas com custo de bateria), on-device vs. híbrido (nuvem — o material observa que os "checkpoints do programa valorizam IA local").
- **Relação com o HigenAI:** 🟢 Aplicável diretamente. Diarização provavelmente ⚪/dispensável (cenário de um profissional por vez). VAD e a escolha streaming/lote e push-to-talk/sempre-escutando são decisões diretamente relevantes caso o HigenAI aceite entrada de voz — e a observação sobre "Hey Meta" não estar disponível é uma restrição técnica concreta que a equipe precisa considerar se pensar em ativação por voz espontânea (hands-free) durante a higienização.

### 5.8 Caminhos de implementação no Android: nativo vs. on-device "de verdade"
- **Conceito:** o Android oferece APIs nativas prontas — `SpeechRecognizer` (STT, pode usar serviço em nuvem por padrão; local garantido via `EXTRA_PREFER_OFFLINE` ou, a partir da API 31, `createOnDeviceSpeechRecognizer()`) e `TextToSpeech` (TTS, inicialização assíncrona, pode ou não usar rede dependendo da voz instalada). Para 100% local e independente do aparelho, os caminhos citados são **whisper.cpp** (Whisper tiny em C++/JNI, ~75MB o modelo tiny) para STT e **Piper via sherpa-onnx** para TTS.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente como roteiro de implementação em camadas — o material sugere explicitamente começar pelas APIs nativas (mais rápidas de integrar) e depois trocar por Whisper tiny/Piper caso o requisito de "IA 100% local" seja um critério de avaliação relevante para o projeto. Fica para a equipe decidir qual nível de "localidade" é necessário para o HigenAI.

As subseções abaixo (5.9 em diante) vêm da **Unidade II** ("Principais abordagens e aplicações"), que é mais introdutória/genérica (notebooks em Google Colab, sem preocupação com celular/edge) — mas cobre as **técnicas de base** de processamento de imagem e de áudio que sustentam o que as Unidades V a XIII descrevem depois para rodar nos óculos.

### 5.9 Visão computacional: catálogo de tarefas (versão introdutória) e casos de uso
- **Conceito:** a Unidade II define visão computacional como a área que permite a máquinas "enxergar" e apresenta a mesma família de tarefas vista na Unidade XII (classificação, detecção, segmentação, OCR), mas com foco em exemplos de aplicação: realidade aumentada, upscaling de imagem, veículos autônomos, análise de imagens médicas (detecção precoce de doenças), reconhecimento facial (Face ID), detecção de emoções por microexpressões.
- **Relação com o HigenAI:** ⚪/🟢 Majoritariamente contexto/motivação geral, mas o exemplo de **análise de imagens médicas para apoiar profissionais de saúde** é uma analogia direta ao domínio do HigenAI (visão computacional aplicada a um contexto clínico/hospitalar).

### 5.10 Processamento de imagem com OpenCV: técnicas de baixo nível
- **Conceito:** um "mini curso" prático com a biblioteca **OpenCV** cobrindo as operações fundamentais sobre pixels: representação digital da imagem (canais de cor BGR/RGB, espaços de cor como HSV), transformações geométricas (espelhamento, rotação), **filtros de suavização/blur** (média, gaussiano, mediana, bilateral — usados para reduzir ruído antes de outras técnicas), **filtro de nitidez** (sharpen/unsharp mask — realça bordas), **detecção de bordas** (Sobel, Canny, Laplaciano — encontra transições bruscas de cor/brilho, base para reconhecimento de formas), e **limiarização/threshold** (binário, binário invertido, adaptativo, Otsu — separa "objeto" de "fundo", útil por exemplo para isolar texto ou uma região de interesse).
- **Relação com o HigenAI:** 🟢 Aplicável diretamente como **camada de pré-processamento**, complementar às tarefas de mais alto nível da Seção 5.1–5.4 (que já vêm prontas em bibliotecas como MediaPipe/ML Kit). Esses fundamentos de OpenCV são relevantes se a equipe precisar: limpar/normalizar o frame antes de alimentar um modelo de landmarks de mãos (blur para reduzir ruído, ajuste de espaço de cor), aplicar um checagem simples de qualidade de frame (ex.: descartar frames borrados/mal iluminados antes de gastar bateria com inferência), ou construir alguma lógica auxiliar própria de visão sem depender de um modelo pronto. O material não indica que o HigenAI precise disso — só descreve as técnicas como parte do repertório do curso.

### 5.11 Classificação de imagem na prática: k-NN vs. CNN (dataset MNIST)
- **Conceito:** exercício prático comparando dois modelos de classificação de dígitos manuscritos (MNIST, 70.000 imagens 28×28 em escala de cinza): **k-Nearest Neighbors (k-NN)** — classifica comparando a imagem nova com as "k" mais parecidas já vistas, simples e sem treinamento complexo — e uma **rede neural convolucional (CNN)** simples. No experimento do material, a CNN teve acurácia de 98,46% contra 96,88% do k-NN. O material também descreve as etapas de **pré-processamento de imagem para treino**: normalização dos valores de pixel (0–255 → 0–1) e ajuste de formato (reshape) para o formato esperado pelo modelo.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente como reforço conceitual (não como recomendação): mostra concretamente que **CNNs superam abordagens mais simples** em uma tarefa de reconhecimento de padrão visual — relevante caso a equipe considere um classificador de gesto/etapa da técnica de higienização como alternativa a landmarks de mãos (ver Seção 5.1). O passo de pré-processamento (normalização + reshape) é genérico e se aplica a qualquer pipeline de classificação de imagem que a equipe venha a montar ou treinar.

### 5.12 YOLO e Whisper: primeiro contato prático (ambiente de nuvem/Colab)
- **Conceito:** a Unidade II também apresenta, de forma introdutória, o **YOLO** (mesma família vista com profundidade na Seção 5.2 — detecção de objetos em uma única passada, com caixas + classe + confiança) e o **Whisper** (transcrição de fala para texto, rodando via `pip install` num notebook Google Colab, com escolha entre modelos tiny/base/small/medium/large e detecção automática de idioma, incluindo português).
- **Relação com o HigenAI:** ⚪ Fundamento teórico/prático introdutório — é a mesma tecnologia da Seção 5.2 (YOLO) e 5.5 (Whisper), mas aqui rodando em nuvem/Colab para fins didáticos, não on-device no celular. Útil como primeiro contato/prototipagem rápida antes de portar para o celular (o próprio material da Unidade XII sugere prototipar a lógica fora do Android antes de integrar), mas não é o caminho de produção para os óculos.

### 5.13 Processamento de áudio e voz: fundamentos do sinal sonoro
- **Conceito:** como um computador "ouve": o áudio captado pelo microfone é amostrado (convertido em sequência de números) e depois processado com técnicas como a **Transformada de Fourier** ou **MFCCs (Mel-Frequency Cepstral Coefficients)** para extrair características relevantes de frequência/tempo — a mesma ideia por trás do "mel-spectrogram" citado na Seção 5.5/5.6. A partir dessa representação numérica, redes neurais aprendem o mapeamento áudio→texto (STT) ou texto→áudio (TTS). O material também lista aplicações do processamento de voz: assistentes virtuais, legendas automáticas (acessibilidade para deficiência auditiva), sintetizadores de voz (para deficiência na fala), dublagens/vozes narrativas em vídeos, tradução simultânea e **biometria de voz** (autenticação por características vocais).
- **Relação com o HigenAI:** ⚪/🟢 A base teórica (amostragem, Fourier, MFCC) é fundamento geral que explica "por dentro" o mel-spectrogram já mapeado como 🟢 na Seção 5.5/5.6. Entre as aplicações citadas, vale destacar duas: **acessibilidade por voz** (se o HigenAI quiser garantir que a resposta sonora seja clara/compreensível, esse é o mesmo princípio usado em sintetizadores de voz assistivos) e **biometria de voz**, que é uma técnica citada no material que *poderia* (não é recomendação) servir para identificar qual profissional está falando/higienizando, algo relacionado ao registro do agente de memória do HigenAI — o material não conecta essa técnica ao registro de identidade, essa é uma leitura do mapeamento, não do curso.

### 5.14 Computação em nuvem, computação de borda (edge) e IoT
- **Conceito:** três modelos de onde o processamento acontece — **nuvem** (poder de processamento remoto, acessível de qualquer lugar, mas depende de rede), **borda/edge** (processamento local, no próprio dispositivo — celular, câmera, smartwatch — mais rápido e menos dependente de internet) e **IoT** (rede de dispositivos conectados que coletam e trocam dados automaticamente). O material usa o exemplo de uma câmera de segurança que reconhece rostos localmente, "economizando tempo e **protegendo a privacidade das pessoas envolvidas**", em vez de enviar o vídeo completo à nuvem.
- **Relação com o HigenAI:** 🟢 Aplicável diretamente — é a mesma lógica de fundo da Seção 4.3 (Edge AI), mas com uma analogia particularmente próxima do HigenAI: uma câmera que reconhece algo sobre uma pessoa (aqui, rostos; no HigenAI, gestos de higienização) e processa **localmente por privacidade**. Como o HigenAI capta imagens de profissionais de saúde (dado sensível, já sinalizado com LGPD na Seção 2.8), esse exemplo reforça — sem recomendar uma escolha específica — por que processar no celular pareado (borda), em vez de enviar vídeo à nuvem, é uma opção que o próprio material associa a privacidade.

---

## Checklist: checkpoints do hackathon × componentes de arquitetura

Esta seção reúne, à parte do restante do mapeamento por conceito, as menções do material a **critérios de avaliação do hackathon ("checkpoints")** e a qual componente da arquitetura cada um corresponde. Vem principalmente da Unidade XIII (tópico 13.2.3.6, "Arquitetura × checkpoints", Tabela 20) e da Unidade XII (tópico 12.10, sobre bateria). Diferente do restante do resumo, isto não é um "conceito" a ser mapeado — é conteúdo prático que a equipe pode usar diretamente para conferir se o desenho do HigenAI cobre o que será avaliado.

| Checkpoint | Componente que satisfaz | Caminho do dado / critério |
|---|---|---|
| Entrada por câmera | Câmera dos óculos + stream do DAT | Óculos → Bluetooth → app companion |
| Entrada por microfone | Microfones dos óculos + áudio HFP | Óculos → Bluetooth → app companion |
| Saída por áudio | Alto-falantes open-ear + áudio Bluetooth do sistema | App companion → Bluetooth → óculos |
| IA local | Modelos on-device no app companion | Inferência inteira no celular, sem nuvem |
| Eficiência de bateria | Uso de foreground service só para trabalho contínuo, WorkManager para trabalho adiável, inferência disparada por gatilho (não em loop cego) | Avaliado pelas decisões de background/energia da Unidade XII (tópico 12.10) |

A Unidade XII também menciona, de forma mais solta (sem uma tabela dedicada como a de checkpoints), que os "checkpoints de IA local e privacidade" valorizam soluções onde o áudio/processamento não sai do aparelho — relevante junto com o ponto de LGPD já mapeado na Seção 2.8.

Vale conferir diretamente o PDF da Unidade 13 (tópico 13.2.3.6) e da Unidade 12 (tópico 12.10) caso a equipe queira o texto completo por trás desta tabela.
