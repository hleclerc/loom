import { defineConfig } from 'vitepress'

export default defineConfig( {
  title: "loom",
  description: "Write a kernel once in C++ or CUDA, call it from Jax, Torch or numpy -- with its gradient.",
  lang: 'en-US',

  // GitHub Pages under github.com/hleclerc/loom
  base: '/loom/',

  cleanUrls: true,
  lastUpdated: true,

  head: [
    [ 'meta', { name: 'theme-color', content: '#0f766e' } ],
  ],

  themeConfig: {
    nav: [
      { text: 'Guide',     link: '/guide/what-is-loom' },
      { text: 'Tutorials', link: '/tutorials/' },
      { text: 'Reference', link: '/reference/python-api' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Getting started',
          items: [
            { text: 'What loom is',           link: '/guide/what-is-loom' },
            { text: 'Why it is not trivial',  link: '/guide/why-not-trivial' },
            { text: 'Installing',             link: '/guide/installing' },
            { text: 'Your first kernel',      link: '/guide/first-kernel' },
          ]
        },
        {
          text: 'Writing kernels',
          items: [
            { text: 'Axes and coordinates',   link: '/guide/axes' },
            { text: 'Roles of the arguments', link: '/guide/arguments' },
            { text: 'The adjoint',            link: '/guide/adjoint' },
            { text: 'Sizes known at run time', link: '/guide/dynamic-sizes' },
          ]
        },
        {
          text: 'Running them',
          items: [
            { text: 'Frameworks and devices',   link: '/guide/frameworks' },
            { text: 'Compilation and catalogue', link: '/guide/compilation' },
          ]
        },
      ],

      '/tutorials/': [
        {
          text: 'Tutorials',
          items: [
            { text: 'All of them',                  link: '/tutorials/' },
            { text: '1 · Diffusion',                link: '/tutorials/diffusion' },
            { text: '2 · Gaussian splatting',       link: '/tutorials/splats' },
          ]
        }
      ],

      '/reference/': [
        {
          text: 'Reference',
          items: [
            { text: 'Python API',            link: '/reference/python-api' },
            { text: 'C++ API',               link: '/reference/cpp-api' },
            { text: 'Environment variables', link: '/reference/environment' },
          ]
        }
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/hleclerc/loom' }
    ],

    search: { provider: 'local' },

    outline: { level: [ 2, 3 ] },

    editLink: {
      pattern: 'https://github.com/hleclerc/loom/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },

    footer: {
      message: 'Status: early. The API may still move.',
      copyright: 'loom — H. Leclerc'
    }
  }
} )
