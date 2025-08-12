'use client';
import { useState, useEffect } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import theme from '../../lib/muiTheme';
import SearchIcon from '@mui/icons-material/Search';
import Autocomplete from '@mui/material/Autocomplete';
import {
  Dialog, DialogTitle, DialogContent, DialogActions,
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
    tagGroups: [['useCases', 'Use Cases'], ['limitations', 'Limitations'], ['risks', 'Risks']] as const,
  },
  llm_apps: {
    label: 'LLM Apps Pages',
    className: 'LLM-Apps Wiki.Code.LLM-Apps WikiClass',
    nameProp: 'llm apps',
    url: 'LLM-Apps Wiki',
    tagGroups: [['useCases', 'Use Cases'], ['limitations', 'Limitations'], ['risks', 'Risks']] as const,
  },
};
type ModeKey = keyof typeof MODES;

const EXTRA_TAG_FIELDS: Record<ModeKey, string[][]> = {
  llms: [['useCases', 'Use Cases'], ['limitations', 'Limitations'], ['risks', 'Risks'], ['modelType', 'Model Type']],
  llm_apps: [['useCases', 'Use Cases'], ['limitations', 'Limitations'], ['risks', 'Risks'], ['llms', 'LLMs Used']],
};

// list for autocomplete
const modelOptions = ['', ''];

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
  
  // keep track of link clicked
  const [linkClicked, setLinkClicked] = useState<string | null>(null);

  // reset tag containers when mode changes (so UI doesn’t show old groups)
  useEffect(() => {
    const tuples = MODES[mode].tagGroups;
    const keys = tuples.map(([k]) => k);
    const groups = Object.fromEntries(keys.map(k => [k, []]));
    
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

      // tuples -> keys
      const keys = tagGroups.map(([k]) => k);

      // grouped sets by key
      const grouped: Record<string, Set<string>> =
        Object.fromEntries(keys.map(k => [k, new Set<string>()]));

      const query = `
        where doc.fullName in (
          select obj.name from BaseObject obj 
          where obj.className = '${className}'
        )
      `.replace(/\s+/g, " ").trim();

      const url = `https://wiki.llm.test/rest/wikis/xwiki/query?q=${encodeURIComponent(query)}&type=xwql&media=json&number=1000&distinct=1&className=${encodeURIComponent(className)}`;

      try {
        const res = await fetch(url, {
          headers: {
            Accept: 'application/json',
            Authorization: 'Basic ' + btoa('ahmed33033:ahmed2003'),
          },
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // populate grouped using KEYS only
        data.searchResults?.forEach((result: any) => {
          const props = result.object?.properties || [];
          props.forEach((p: any) => {
            const key = p.name?.trim();        // e.g. "useCases"
            const val = p.value;               // comma-separated string
            if (!key || !grouped[key] || typeof val !== 'string') return;
            val.split(',').forEach((tag: string) => {
              const t = tag.trim().toLowerCase();
              if (t) grouped[key].add(t);
            });
          });
        });

        // flatten to arrays for UI
        const flat: TagMap = {};
        for (const [key] of tagGroups) {
          flat[key] = Array.from(grouped[key] ?? new Set<string>()).sort((a, b) => a.localeCompare(b));
        }
        setAvailableTags(flat);
      } catch (err) {
        console.error('Failed to fetch tags:', err);
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

    const fullUrl = "https://wiki.llm.test/rest/wikis/xwiki/query" +
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
    } finally {
      setDidSearch(true);
      setLinkClicked(null);
    }
  };

  // ---- Filters dialog helpers ----
  const openFilters = () => setFiltersOpen(true);
  const closeFilters = () => setFiltersOpen(false);
  const clearGroup = (group: string) =>
    setSelectedTags(prev => ({ ...prev, [group]: [] }));
  const clearAll = () => setSelectedTags({});
  const tagTuples = MODES[mode].tagGroups;
  const keyToLabel = Object.fromEntries(tagTuples);

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

            <Autocomplete
              options={modelOptions}
              freeSolo
              inputValue={searchTerm}
              onInputChange={(_, value) => setSearchTerm(value)}
              // when a user picks an option, update and run the search
              onChange={(_, value) => {
                if (value != null) {
                  setSearchTerm(value);
                  searchPages();
                }
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') { e.preventDefault(); searchPages(); }
              }}

              // Fill the space where your InputBase was
              sx={{ ml: 1, flex: 1, minWidth: 0 }}

              // render InputBase
              renderInput={(params) => (
                <Box sx={{ position: 'relative', width: '100%' }}>
                  <InputBase
                    placeholder={`Search by ${MODES[mode].nameProp}`}
                    fullWidth
                    inputRef={params.InputProps.ref}
                    inputProps={params.inputProps}
                    sx={{}}
                  />
                  {/* shows the clear button & dropdown icon */}
                  <Box
                    sx={{
                      position: 'absolute',
                      top: 0,
                      right: 0,
                      height: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      pr: 0.5,
                    }}
                  >
                    {params.InputProps.endAdornment}
                  </Box>
                </Box>
              )}
            />
            <IconButton onClick={searchPages}><SearchIcon /></IconButton>
          </Paper>

          <Button variant="outlined" onClick={openFilters}>Filters</Button>
          <Button variant="contained" onClick={searchPages}>Search</Button>
        </Box>

        {/* Selected Filter Chips */}
        <Box sx={{ mt: 2 }}>
          {Object.entries(selectedTags).flatMap(([groupKey, tags]) =>
            (tags || []).map((tag) => (
              <Chip
                key={`${groupKey}-${tag}`}
                label={`${keyToLabel[groupKey] ?? groupKey}: ${tag}`}
                onDelete={() => handleToggle(groupKey, tag)}  // live toggle
                sx={{ mr: 1, mb: 1 }}
              />
            ))
          )}
        </Box>

        {/* Error Message */}
        {error && <pre style={{ color: 'red' }}>{error}</pre>}

        {/* Search Results */}
          {/* display the xwiki page using an iframe*/}
        {linkClicked !== null ? (
          <div className="w-full flex-1 flex justify-center">
            <iframe
              src={linkClicked}
              title={`Wiki: ${linkClicked}`}
              className="w-full max-w-7xl h-[80vh] rounded-xl border-2 border-[#2D2A5A] bg-white"
              style={{ minHeight: 400 }}
            />
          </div>
        ) : /* display the search results otherwise */ (
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 2 }}>
            {results.length === 0 ? ( // check if the user did a search with results
              didSearch === false || error !== null ? (
                <Typography variant="body2" sx={{ opacity: 0.8 }}></Typography>
                ) : (
                <Typography variant="body2" sx={{ opacity: 0.8 }}>No results found.</Typography>)
            ) : (
              results.map((r, index) => {
                const props = propsToMap(r);

                // Name / title
                const name =
                  r.title ||
                  props.name ||
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
                const tagTuples = MODES[mode].tagGroups;
                const extraTuples = EXTRA_TAG_FIELDS[mode];
                const keyToLabel = Object.fromEntries([...tagTuples, ...extraTuples]);
                const tagFields = Array.from(new Set([
                  ...tagTuples.map(([k]) => k),
                  ...extraTuples.map(([k]) => k),
                ]));

                const chips = tagFields.flatMap((field) =>
                  splitTags(String(props[field] ?? '')).map((t) => ({ field, t }))
                );

                const chipsByField = chips.reduce((acc, c) => {
                  (acc[c.field] ??= []).push(c);
                  return acc;
                }, {} as Record<string, { field: string; t: string }[]>);

                // view URL
                const appName = name;
                const encodedName = encodeURIComponent(String(appName).trim());

                // display box for each result
                return (
                  <Box 
                  key={index} 
                  sx={{ 
                    border: '1px solid', borderColor: 'divider', borderRadius: 1, p: 2,
                    cursor: 'pointer', '&:hover': {backgroundColor: 'action.hover', boxShadow: 2,
                  }}}
                  onClick={() => setLinkClicked(`https://wiki.llm.test/bin/view/${MODES[mode].url}/${encodedName}`)}>
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
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                        {Object.entries(chipsByField).map(([groupKey, items]) => (
                          <Box key={groupKey} sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 0.75 }}>
                            <Typography variant="caption" sx={{ mr: 0.5 }}>
                              {keyToLabel[groupKey] ?? groupKey}:
                            </Typography>
                            {items.map(({ field, t }, i) => (
                              <Chip key={`${field}-${t}-${i}`} label={t} size="small" />
                            ))}
                          </Box>
                        ))}
                      </Box>
                    )}
                  </Box>
                );
              })
            )}
          </Box>
        )}
      </Box>

      {/* Filters Dialog */}
      <Dialog open={filtersOpen} onClose={closeFilters} fullWidth maxWidth="md">
        <DialogTitle>Filters</DialogTitle>
        <DialogContent dividers>
          {/* Live-selected chips */}
          <Box sx={{ mb: 2 }}>
            {Object.entries(selectedTags).flatMap(([groupKey, tags]) =>
              (tags || []).map((tag) => (
                <Chip
                  key={`${groupKey}-${tag}`}
                  label={`${keyToLabel[groupKey] ?? groupKey}: ${tag}`}
                  onDelete={() => handleToggle(groupKey, tag)}  // live toggle
                  sx={{ mr: 1, mb: 1 }}
                />
              ))
            )}
          </Box>

          {/* Groups (iterate tuples so we keep order + have labels) */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            {tagTuples.map(([key, label]) => {
              const values = availableTags[key] ?? [];
              const selected = new Set(selectedTags[key] ?? []);
              return (
                <Box
                  key={key}
                  sx={{
                    width: 260,
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 1,
                    p: 1.5,
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                    <Typography sx={{ fontWeight: 600 }}>{label}</Typography>
                    <Button size="small" onClick={() => clearGroup(key)}>Clear</Button>
                  </Box>

                  <Box sx={{ maxHeight: 240, overflowY: 'auto', pr: 1 }}>
                    <FormGroup>
                      {values.map((tag) => (
                        <FormControlLabel
                          key={tag}
                          control={
                            <Checkbox
                              checked={selected.has(tag)}
                              onChange={() => handleToggle(key, tag)}
                            />
                          }
                          label={tag}
                        />
                      ))}
                    </FormGroup>
                  </Box>
                </Box>
              );
            })}
          </Box>
        </DialogContent>

        <DialogActions>
          <Button onClick={clearAll}>Clear All</Button>
          <Box sx={{ flexGrow: 1 }} />
          <Button onClick={closeFilters}>Close</Button>
        </DialogActions>
      </Dialog>
    </ThemeProvider>
  );
}
