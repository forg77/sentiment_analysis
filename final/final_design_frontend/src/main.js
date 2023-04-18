import { createApp } from 'vue';
import homePage from "@/views/home-page";
import searchPage from "@/views/search-page";
import App from './App'
import axios from "axios";
import ElementPlus from 'element-plus';
import 'element-plus/dist/index.css';
axios.defaults.baseURL = '/api'
import {createRouter, createWebHistory} from "vue-router/dist/vue-router";
const app = createApp(App);
app.use(ElementPlus);
const router = createRouter({
    history: createWebHistory(),
    routes:[
        {
            path:'/',
            name:'homepage',
            component: homePage
        },
        {
            path:'/api/search',
            name:'search',
            component: searchPage
        }

    ]
})
app.use(router);
app.mount('#app');
