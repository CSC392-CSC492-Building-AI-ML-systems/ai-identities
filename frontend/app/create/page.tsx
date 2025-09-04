"use client";

import { useParams } from "next/navigation";
import { XWIKI_URL } from "@/constants";
import { useState } from "react";
import {
  Button,
  CssBaseline,
  InputLabel,
  TextField,
  ThemeProvider,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import theme from "@/lib/muiTheme";
import { useXWikiAuth } from "@/hooks/useXWikiAuth";

type ModeKey = "llms" | "llm_apps";
export default function WikiIframePage() {
  const [pageName, setPageName] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const { loggedIn, loading } = useXWikiAuth();

  const [mode, setMode] = useState<ModeKey>("llms");
  if (loading) {
    return <div></div>;
  }

  if (!loggedIn) {
    window.location.href = `${XWIKI_URL}/bin/login/XWiki/XWikiLogin?xredirect=${XWIKI_URL.split("://").join(
      "%3A%2F%2F"
    )}%2Fbin%2Fview%2Fredir%3Fnext%3D%2Fcreate`;
    return <div></div>;
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <main className=" bg-[#050a1f] mt-56 flex flex-col items-center w-screen">
        {!submitted ? (
          <div style={{ display: "flex", flexDirection: "row", columnGap: "50px" }}>
            <ToggleButtonGroup
              size="small"
              exclusive
              value={mode}
              sx={{
                border: "1px solid",
                borderColor: "divider",
                borderRadius: 1,
                overflow: "hidden",
                "& .MuiToggleButton-root": {
                  border: 0,
                  px: 1.25,
                  py: 0.25,
                  fontSize: 12,
                },
                "& .Mui-selected": { bgcolor: "action.selected" },
              }}
              onChange={(_, v) => {
                if (v) {
                  setMode(v as ModeKey);
                }
              }}
            >
              <ToggleButton value="llms">LLMs</ToggleButton>
              <ToggleButton value="llm_apps">Apps</ToggleButton>
            </ToggleButtonGroup>

            <TextField
              label="Page Name"
              value={pageName}
              onChange={(e) => {
                setPageName(e.target.value);
              }}
            />
            <Button
              onClick={() => {
                setSubmitted(true);
              }}
            >
              Create page
            </Button>
          </div>
        ) : (
          <div className="flex-1 flex justify-center w-screen absolute bottom-0 h-[90vh]">
            <iframe
              src={XWIKI_URL + (mode === "llms" ? "/bin/view/XWiki/LLMClass" : "/bin/view/XWiki/LLMAppClass")}
              title="Create Wiki Page"
              className="w-97/100 h-[90vh]"
              id="createFrame"
              name="createFrame"
              onLoad={() => {
                const iframe = document.getElementById("createFrame") as HTMLIFrameElement | null;

                console.log(iframe);
                if (iframe) {
                  iframe.contentWindow?.postMessage(pageName, XWIKI_URL);
                  iframe.className = "w-full";
                }
              }}
            />
          </div>
        )}
      </main>
    </ThemeProvider>
  );
}
