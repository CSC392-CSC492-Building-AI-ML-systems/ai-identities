'use client';
import { useState, useEffect } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import theme from '../../lib/muiTheme';
import {
  TextField,
  Button,
  Menu,
  MenuItem,
  Checkbox,
  ListItemText,
  IconButton,
  Box
} from '@mui/material';
import FilterListIcon from '@mui/icons-material/FilterList';
import { resolve } from 'node:dns';

export default function HomePage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTags, setSelectedTags] = useState<{ [group: string]: string[] }>({});
  const [availableTags, setAvailableTags] = useState<{ [group: string]: string[] }>({
    useCases: [],
    limitations: [],
    risks: [],
  });
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const open = Boolean(anchorEl);

  // Load tags from XWiki
  useEffect(() => {
    const fetchTags = async () => {
      const query = `
        where doc.fullName in (
          select obj.name from BaseObject obj 
          where obj.className = 'LLM Wiki.Code.LLM WikiClass'
        )
      `.replace(/\s+/g, " ").trim();

      const url = `http://159.203.20.200:8080/rest/wikis/xwiki/query?q=${encodeURIComponent(query)}&type=xwql&media=json&number=1000&distinct=1&className=${encodeURIComponent('LLM Wiki.Code.LLM WikiClass')}`;

      try {
        const res = await fetch(url, { headers: { 'Accept': 'application/json', "Authorization": "Basic " + btoa("ahmed33033:ahmed2003") } });
        const data = await res.json();
        const groupedTags: { [key: string]: Set<string> } = {
          useCases: new Set(),
          limitations: new Set(),
          risks: new Set()
        };

        data.searchResults.forEach((result: any) => {
          const object = result.object;
          if (!object?.properties) return;

          object.properties.forEach((prop: any) => {
            const key = prop.name.trim();
            const val = prop.value;

            if (!groupedTags[key] || typeof val !== 'string') return;

            val.split(',').forEach((tag: string) => {
              const trimmed = tag.trim().toLowerCase();
              if (trimmed) groupedTags[key].add(trimmed);
            });
          });
        });

        const flatTags: { [group: string]: string[] } = {};
        for (const group in groupedTags) {
          flatTags[group] = Array.from(groupedTags[group]).sort();
        }

        setAvailableTags(flatTags);
        console.log("HHHHHHHHHHHHHHHHHHHHH", flatTags)
      } catch (err) {
        console.error("Failed to fetch tags:", err);
      }
    };

    fetchTags();
  }, []);

  const handleToggle = (group: string, value: string) => {
    setSelectedTags((prev) => {
      const current = prev[group] || [];
      const newGroup = current.includes(value)
        ? current.filter((v) => v !== value)
        : [...current, value];

      return { ...prev, [group]: newGroup };
    });
  };

  const searchPages = async () => {
    let joins = "select obj.name from BaseObject obj";
    const conditions = ["obj.className = 'LLM Wiki.Code.LLM WikiClass'"];
    let spIndex = 1;

    for (const group in selectedTags) {
      for (const value of selectedTags[group]) {
        const alias = `sp${spIndex++}`;
        joins += `, StringProperty ${alias}`;
        conditions.push(
          `obj.id = ${alias}.id.id`,
          `${alias}.name = '${group}'`,
          `lower(${alias}.value) like '%${value.replace(/'/g, "''")}%'`
        );
      }
    }

    let query = `where doc.fullName in (${joins} where ${conditions.join(" and ")})`;

    if (searchTerm) {
      query += ` and doc.fullName in (
        select objn.name from BaseObject objn, StringProperty spn
        where objn.className = 'LLM Wiki.Code.LLM WikiClass'
        and objn.id = spn.id.id
        and spn.name = 'name'
        and lower(spn.value) like '%${searchTerm.replace(/'/g, "''").toLowerCase()}%'
      )`;
    }

    query += " and doc.fullName <> 'LLM Wiki.Code.LLM WikiTemplate'";
    query = query.replace(/\s+/g, " ").trim();

    const fullUrl = "http://159.203.20.200:8080/rest/wikis/xwiki/query" +
      "?q=" + encodeURIComponent(query) +
      "&type=xwql&media=json&number=100";

    try {
      const res = await fetch(fullUrl, { headers: { "Accept": "application/json", "Authorization": "Basic " + btoa("ahmed33033:ahmed2003")} });
      const data = await res.json();
      console.log("SEARCH RESULT: ", res);
      console.log("SEARCH DATA: ", data);
      setResults(data.searchResults || []);
      setError(null);
    } catch (err: any) {
      setError("Error: " + err.message);
      setResults([]);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

        <Box sx={{ maxWidth: 800, margin: '2rem auto', padding: '1rem', paddingTop: '80px' }}>
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <TextField
            label="Search by page name"
            variant="outlined"
            fullWidth
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            />

            <IconButton onClick={(e) => setAnchorEl(e.currentTarget)}>
            <FilterListIcon />
            </IconButton>

            <Button variant="contained" onClick={searchPages}>
            Search
            </Button>
        </Box>

        <Menu anchorEl={anchorEl} open={open} onClose={() => setAnchorEl(null)}>
            {Object.entries(availableTags).map(([group, values]) => (
                <Box key={group} sx={{ px: 2, py: 1 }}>
                <strong style={{ textTransform: 'capitalize' }}>{group}</strong>
                {Array.from(values).sort().map((tag) => (
                    <MenuItem key={tag} onClick={() => handleToggle(group, tag)}>
                    <Checkbox checked={selectedTags[group]?.includes(tag) || false} />
                    <ListItemText primary={tag} />
                    </MenuItem>
                ))}
                </Box>
            ))}
        </Menu>

        {error && <pre style={{ color: 'red' }}>{error}</pre>}

        <ul>
            {results.length === 0 ? (
                <li>No results found.</li>
            ) : (
                results.map((r, index) => {
                const space = r.space || "";
                const pageName = r.pageName || "";
                const title = r.title || pageName || space;

                // Convert space like "LLM Wiki.Claude Opus 4" → "LLM Wiki/Claude Opus 4"
                const spacePath = space.replace(/\./g, "/");
                const appName = r.title || r.pageName || r.space;
                const encodedName = encodeURIComponent(appName.trim());
                const viewUrl = `/wiki/LLM Wiki/${encodedName}`;

                return (
                    <li key={index}>
                    <a
                        href={viewUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ color: "#F3F3FF", textDecoration: "underline" }}
                    >
                        {title}
                    </a>
                    </li>
                );
                })
            )}
        </ul>
        </Box>
    </ThemeProvider>
  );
}
