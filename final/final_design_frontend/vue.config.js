module.exports = {
  devServer:{
    proxy:{
      '/search':{
        target: 'http://localhost:12345',
        changeOrigin: true,
      },

    }
  }
}
