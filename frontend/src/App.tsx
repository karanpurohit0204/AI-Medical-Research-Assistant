import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { AskPage } from './pages/AskPageFixed'
import { PapersPage } from './pages/PapersPage'
export default function App() { return <BrowserRouter><Routes><Route element={<Layout/>}><Route index element={<AskPage/>}/><Route path="papers" element={<PapersPage/>}/></Route></Routes></BrowserRouter> }
