'use client';

import { useParams } from 'next/navigation';
import { XWIKI_URL } from '@/constants';
import { useState } from 'react';
import { CssBaseline, InputLabel, TextField, ThemeProvider } from '@mui/material';
import theme from '@/lib/muiTheme';
import { useXWikiAuth } from '@/hooks/useXWikiAuth';
import { useRouter } from 'next/navigation'
export default function WikiIframePage() {
    const params = useParams();
    const [firstLoad, setFirstLoad] = useState(true)
    const [pageName, addPageName] = useState('')
    const [submitted, setSubmitted] = useState(false)

    const {loggedIn, loading} = useXWikiAuth();

    const router = useRouter()
    if(loading){
        return <div></div>
    }

    if(!loggedIn){
        window.location.href = "https://wiki.llm.test/bin/login/XWiki/XWikiLogin?xredirect=https%3A%2F%2Fwiki.llm.test%2Fbin%2Fview%2Fredir%3Fnext%3D%2Fcreate"
        return <div></div>
    }
    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <main className="min-h-screen bg-[#050a1f] mt-56 flex flex-col items-center w-screen">
                {!submitted ?
                    <div>
                        <TextField label="Page Name" value={pageName} />    
                        <button onClick={()=>{setSubmitted(true)}}></button>
                    </div>
                    :
                    <div className="flex-1 flex justify-center w-screen">
                        <iframe
                            src={XWIKI_URL + "/bin/view/XWiki/LLMClass"}
                            className="w-full hidden"
                            id="createFrame"
                            name="createFrame"
                            onLoad={() => {
                                const iframe = (document.getElementById('createFrame') as HTMLIFrameElement | null);

                                console.log(iframe)
                                if (iframe) {
                                    iframe.contentWindow?.postMessage(pageName, "*")
                                    iframe.className = "w-full"
                                }
                            }}
                            style={{ minHeight: 400 }}
                        />
                    </div>
                }
            </main>
        </ThemeProvider>
    );
}