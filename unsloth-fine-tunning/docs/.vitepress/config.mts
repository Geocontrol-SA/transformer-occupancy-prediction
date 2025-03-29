import { defineConfig } from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Bus Occupancy Prediction",
  description: "A System of Bus Occupancy Prediction using GPT networks",
  srcDir: ".", // Define a raiz do projeto como o diretório de origem
  // exclude: ['node_modules/**', 'dist/**'], // Ignora diretórios desnecessários
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: "Home", link: "/" },
      // { text: "Examples", link: "/markdown-examples" },
    ],

    sidebar: [
      {
        text: "Fine tune",
        items: [
          { text: "How to make a Fine tune", link: "/fine_tuning" },
          {
            text: "Fine tune with notebook",
            link: "/fine_tuning_with_notebook",
          },
        ],
      },
      {
        text: "Predictions",
        items: [
          {
            text: "How to make a prediction without notebook",
            link: "/fine_tuning#running-prediction-with-mistral-example",
          },
        ],
      },
      {
        text: "Article",
        link: "/article.claude",
        // items: [
        //   {
        //     text: "Article",

        //   },
        // ],
      },
    ],

    socialLinks: [{ icon: "github", link: "https://github.com/btoffoli" }],
  },
});
