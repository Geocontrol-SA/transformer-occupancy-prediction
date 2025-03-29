import os


static_content = '''
### Running complete documentation as a blog

- Using npm, pnpm or yarn

```bash
#Resolve depenencies
npm -i
```

```bash
#Running blog
npm docs:dev
'''

docs_path = 'docs'
readme_path = 'README.md'

# Listar arquivos .md na pasta docs
md_files = [f for f in os.listdir(docs_path) if f.endswith('.md') if not f.startswith('index')]

# Criar a lista de links
links = "\n".join([f"- [{os.path.splitext(f)[0].replace('-', ' ').title()}]({os.path.join(docs_path, f)})" for f in md_files])

# Ler o README.md existente
with open(readme_path, 'r', encoding='utf-8') as file:
    readme_content = file.read()

# Substituir a seção de documentação
new_content = readme_content.split('## Documentação')[0]
new_content += '## Documentação\n\n' + links

# Escrever de volta no README.md
with open(readme_path, 'w', encoding='utf-8') as file:
    file.write(f"{new_content}\n\n{static_content}")

print("Menu atualizado no README.md!")
