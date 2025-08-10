'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import theme from '../../lib/muiTheme';
import SearchIcon from '@mui/icons-material/Search';
import {
  Dialog, DialogTitle, DialogContent, DialogActions,
  Select, MenuItem,
  Typography, FormGroup, FormControlLabel, Checkbox, Chip,
  Button, Box, Paper, InputBase, IconButton, ToggleButtonGroup, ToggleButton
} from '@mui/material';

type TagMap = { [group: string]: string[] };

const MODES = {
  llms: {
    label: 'LLM Pages',
    className: 'LLM Wiki.Code.LLM WikiClass',
    nameProp: 'llms',
    url: 'LLM Wiki',
    tagGroups: ['useCases', 'limitations', 'risks'] as const,
  },
  llm_apps: {
    label: 'LLM Apps Pages',
    className: 'LLM-Apps Wiki.Code.LLM-Apps WikiClass',
    nameProp: 'llm apps',
    url: 'LLM-Apps Wiki',
    tagGroups: ['useCases', 'limitations', 'risks'] as const,
  },
};
type ModeKey = keyof typeof MODES;

const EXTRA_TAG_FIELDS: Record<ModeKey, string[]> = {
  llms: ['useCases', 'limitations', 'risks', 'modelType'],
  llm_apps: ['useCases', 'limitations', 'risks', 'llms', 'llmsGuess'],
};

// helper functions to display search results
const propsToMap = (r: any) => {
  const map: Record<string, any> = {};
  (r.object?.properties || []).forEach((p: any) => {
    map[p.name?.trim()] = p.value;
  });
  return map;
};

const splitTags = (s?: string) =>
  (s || '')
    .split(',')
    .map(t => t.trim().toLowerCase())
    .filter(Boolean);

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

  // Dialog state for filters
  const [filtersOpen, setFiltersOpen] = useState(false);

  // add state
  const [mode, setMode] = useState<ModeKey>('llms');
  const [didSearch, setDidSearch] = useState<Boolean>(false);

  // reset tag containers when mode changes (so UI doesn’t show old groups)
  useEffect(() => {
    const groups = Object.fromEntries(MODES[mode].tagGroups.map(g => [g, []]));
    setSelectedTags(groups as TagMap);
    setAvailableTags(groups as TagMap);

    setResults([]);
    setError(null);

    setSearchTerm('');
    setDidSearch(false);
  }, [mode]);

  // Load tags from XWiki
  useEffect(() => {
    const fetchTags = async () => {
      const { className, tagGroups } = MODES[mode];

      const query = `
        where doc.fullName in (
          select obj.name from BaseObject obj 
          where obj.className = '${className}'
        )
      `.replace(/\s+/g, " ").trim();

      const url = `http://159.203.20.200:8080/rest/wikis/xwiki/query?q=${encodeURIComponent(query)}&type=xwql&media=json&number=1000&distinct=1&className=${encodeURIComponent(className)}`;

      try {
        const res = await fetch(url, { headers: { 'Accept': 'application/json', "Authorization": "Basic " + btoa('ahmed33033:ahmed2003') } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        const grouped: { [key: string]: Set<string> } =
          Object.fromEntries(tagGroups.map(g => [g, new Set<string>()]));

        data.searchResults?.forEach((result: any) => {
          const props = result.object?.properties || [];
          props.forEach((p: any) => {
            const key = p.name?.trim();
            const val = p.value;
            if (!key || !grouped[key] || typeof val !== 'string') return;
            val.split(',').forEach((tag: string) => {
              const t = tag.trim().toLowerCase();
              if (t) grouped[key].add(t);
            });
          });
        });

        const flat: TagMap = {};
        for (const g of tagGroups) flat[g] = Array.from(grouped[g] || []).sort();
        setAvailableTags(flat);
      } catch (err) {
        console.error("Failed to fetch tags:", err);
      }
    };

    fetchTags();
  }, [mode]);

  const handleToggle = (group: string, value: string) => {
    setSelectedTags((prev) => {
      const current = prev[group] || [];
      const next = current.includes(value)
        ? current.filter((v) => v !== value)
        : [...current, value];
      return { ...prev, [group]: next };
    });
  };
  /* ------------------------------
              SEARCH
  ------------------------------ */
  const searchPages = async () => {
    const { className} = MODES[mode];

    let joins = "select obj.name from BaseObject obj";
    const conditions = [`obj.className = '${className}'`];
    let spIndex = 1;

    for (const group in selectedTags) {
      for (const value of selectedTags[group] || []) {
        const alias = `sp${spIndex++}`;
        joins += `, StringProperty ${alias}`;
        conditions.push(
          `obj.id = ${alias}.id.id`,
          `${alias}.name = '${group}'`,
          `lower(${alias}.value) like '%${value.replace(/'/g, "''").toLowerCase()}%'`
        );
      }
    }

    let query = `where doc.fullName in (${joins} where ${conditions.join(" and ")})`;

    if (searchTerm) {
      query += ` and doc.fullName in (
        select objn.name from BaseObject objn, StringProperty spn
        where objn.className = '${className}'
        and objn.id = spn.id.id
        and spn.name = 'name'
        and lower(spn.value) like '%${searchTerm.replace(/'/g, "''").toLowerCase()}%'
      )`;
    }

    query += " and doc.fullName <> 'LLM Wiki.Code.LLM WikiTemplate'";
    query += " and doc.fullName <> 'LLM-Apps Wiki.Code.LLM-Apps WikiTemplate'";
    query = query.replace(/\s+/g, " ").trim();

    const fullUrl = "http://159.203.20.200:8080/rest/wikis/xwiki/query" +
      "?q=" + encodeURIComponent(query) +
      `&type=xwql&media=json&number=100&className=${encodeURIComponent(className)}`;

    try {
      const res = await fetch(fullUrl, { headers: { "Accept": "application/json", "Authorization": "Basic " + btoa("ahmed33033:ahmed2003")} });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      console.log("SEARCH QUERY: ", fullUrl);
      console.log("SEARCH DATA: ", data);
      setResults(data.searchResults || []);
      setDidSearch(true);
      setError(null);
    } catch (err: any) {
      setError("Failed to fetch search data: " + (err?.message || String(err)));
      setResults([]);
    }
  };

  // ---- Filters dialog helpers ----
  const openFilters = () => setFiltersOpen(true);
  const closeFilters = () => setFiltersOpen(false);
  const clearGroup = (group: string) =>
    setSelectedTags(prev => ({ ...prev, [group]: [] }));
  const clearAll = () => setSelectedTags({});

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      <Box sx={{ maxWidth: 1200, margin: '2rem auto', padding: '1rem', paddingTop: '80px' }}>
        {/* Search Bar + Filters Button */}
        <Box sx={{ display: 'flex', gap: 2, mb: 2, alignItems: 'center', justifyContent: 'center',}}>
          <Paper variant="outlined" sx={{ width: '60%', display: 'flex', alignItems: 'center', justifyContent: 'flex-start', px: 2, py: 0.5, gap: 1 }}>
            <ToggleButtonGroup size="small" exclusive value={mode}
              sx={{
                border: '1px solid', borderColor: 'divider', borderRadius: 1, overflow: 'hidden',
                '& .MuiToggleButton-root': { border: 0, px: 1.25, py: 0.25, fontSize: 12 },
                '& .Mui-selected': { bgcolor: 'action.selected' },
              }}
              onChange={(_, v) => { if (v) { setMode(v as ModeKey); setResults([]); setError(null); } }}
            >
              <ToggleButton value="llms">LLMs</ToggleButton>
              <ToggleButton value="llm_apps">Apps</ToggleButton>
            </ToggleButtonGroup>

            <InputBase
              onKeyDown={(e) => {
                if (e.key === 'Enter') { e.preventDefault(); searchPages(); }
              }}
              sx={{ ml: 1, flex: 1 }}
              placeholder={`Search by ${MODES[mode].nameProp}`}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <IconButton onClick={searchPages}><SearchIcon /></IconButton>
          </Paper>

          <Button variant="outlined" onClick={openFilters}>Filters</Button>
          <Button variant="contained" onClick={searchPages}>Search</Button>
        </Box>

        {/* Selected Filter Chips */}
        <Box sx={{ mt: 2 }}>
          {Object.entries(selectedTags).flatMap(([group, tags]) =>
            (tags || []).map((tag) => (
              <Chip
                key={`${group}-${tag}`}
                label={`${group}: ${tag}`}
                onDelete={() => handleToggle(group, tag)}
                sx={{ mr: 1, mb: 1 }}
              />
            ))
          )}
        </Box>

        {/* Error Message */}
        {error && <pre style={{ color: 'red' }}>{error}</pre>}

        {/* Search Results */}
        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 2 }}>
          {results.length === 0 ? (
            didSearch === false || error !== null ? (
              <Typography variant="body2" sx={{ opacity: 0.8 }}></Typography>
              ) : (
              <Typography variant="body2" sx={{ opacity: 0.8 }}>No results found.</Typography>)
          ) : (
            results.map((r, index) => {
              const props = propsToMap(r);

              // Name / title
              const name =
                props.name ||
                r.title ||
                r.pageName ||
                r.space ||
                'Untitled';

              // Creator + Release date
              const creator = props.creatorName || 'Unknown';
              const release =
                props.releaseDate
                  ? new Date(props.releaseDate).toLocaleDateString()
                  : undefined;

              // collect tag chips (groups + a couple mode-specific extras)
              const tagFields = Array.from(
                new Set([...(MODES[mode].tagGroups as readonly string[]), ...EXTRA_TAG_FIELDS[mode]])
              );

              const chips = tagFields.flatMap((field) =>
                splitTags(String(props[field] ?? '')).map((t) => ({ field, t }))
              );

              // view URL
              const appName = name;
              const encodedName = encodeURIComponent(String(appName).trim());
              const viewUrl = `/wiki/${MODES[mode].url}/${encodedName}`;

              return (
                <Box key={index} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1, p: 2 }}>
                  <Link href={viewUrl}>
                    <Box sx={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="h6" sx={{ mr: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {name}
                      </Typography>
                    </Box>

                    <Typography variant="body2" sx={{ mb: 1.5 }}>
                      Creator: <strong>{creator}</strong>
                      {release ? <> • Released: <strong>{release}</strong></> : null}
                    </Typography>

                    {chips.length > 0 && (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                        {chips.map(({ field, t }, i) => (
                          <Chip key={`${field}-${t}-${i}`} label={`${field}: ${t}`} size="small" />
                        ))}
                      </Box>
                    )}
                  </Link>
                </Box>
              );
            })
          )}
        </Box>
      </Box>

      {/* Filters Dialog */}
      <Dialog open={filtersOpen} onClose={closeFilters} fullWidth maxWidth="md">
        <DialogTitle>Filters</DialogTitle>
        <DialogContent dividers>
          {/* Live-selected chips */}
          <Box sx={{ mb: 2 }}>
            {Object.entries(selectedTags).flatMap(([group, tags]) =>
              (tags || []).map((tag) => (
                <Chip
                  key={`${group}-${tag}`}
                  label={`${group}: ${tag}`}
                  onDelete={() => handleToggle(group, tag)}  // live toggle
                  sx={{ mr: 1, mb: 1 }}
                />
              ))
            )}
          </Box>

          {/* Groups */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            {Object.entries(availableTags).map(([group, values]) => (
              <Box
                key={group}
                sx={{
                  width: 260,
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                  p: 1.5,
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                  <Typography sx={{ textTransform: 'capitalize', fontWeight: 600 }}>{group}</Typography>
                  <Button size="small" onClick={() => clearGroup(group)}>Clear</Button>
                </Box>

                <Box sx={{ maxHeight: 240, overflowY: 'auto', pr: 1 }}>
                  <FormGroup>
                    {values.map((tag) => (
                      <FormControlLabel
                        key={tag}
                        control={
                          <Checkbox
                            checked={selectedTags[group]?.includes(tag) || false}
                            onChange={() => handleToggle(group, tag)}  // live toggle
                          />
                        }
                        label={tag}
                      />
                    ))}
                  </FormGroup>
                </Box>
              </Box>
            ))}
          </Box>
        </DialogContent>

        {/* Optional tiny footer: just Close + Clear All */}
        {/* Remove this whole block if you want literally no buttons */}
        <DialogActions>
          <Button onClick={clearAll}>Clear All</Button>
          <Box sx={{ flexGrow: 1 }} />
          <Button onClick={closeFilters}>Close</Button>
        </DialogActions>
      </Dialog>
    </ThemeProvider>
  );
}
