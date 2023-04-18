<template>
  <el-card class="box-card">
    <div>这个项目是一个可以对特定B站视频做分析的项目。实现了情感倾向分析和对高强度词做词云展示的功能</div>
    <div>使用了macbert+full connect做分类器 https://github.com/ymcui/MacBERT</div>
    <div>情感词典在dutir情感本体库的基础上添加了一些新词 https://ir.dlut.edu.cn/zyxz/qgbtk.htm</div>
    <div>详细的代码可以访问https://github.com/forg77/sentiment_analysis</div>
  </el-card>
  <div class="center-container">
    <el-input class="center-input" v-model="input" placeholder="请输入一个bv号"></el-input>
<!--    <el-button icon="el-icon-search" circle @click="handleSearch" ></el-button>-->
    <el-button plain type="primary" @click="handleSearch" size="medium">GO</el-button>
    <!--    <input type="text" v-model="input" />-->
<!--    <button @click="handleSearch">Search</button>-->
  </div>


</template>
<style>
.center-container{
  display: flex;
  justify-content: center;
  align-items: center;
  height: 30vh;
}
.center-input {
  max-width: 90%;
  margin: 0 auto;
}

</style>
<script>
export default {
  data() {
    return {
      input : ''
    }
  },
  methods: {
    handleSearch() {
      // 处理用户输入内容
      this.$router.push(`/api/search?q=${this.input}`)
    }
  }

}
</script>
