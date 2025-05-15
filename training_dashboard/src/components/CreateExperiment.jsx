import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  CircularProgress,
  Alert,
  Slider,
  Switch,
  FormControlLabel
} from '@mui/material';

const CreateExperiment = ({ onExperimentCreated }) => {
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [creating, setCreating] = useState(false);
  
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    algorithm: 'dpo',
    policy_model_id: '',
    reference_model_id: '',
    reward_model_id: '',
    dataset_id: '',
    batch_size: 4,
    learning_rate: 0.00001,
    max_steps: 1000,
    seed: 42,
    algorithm_config: {
      beta: 0.1,  // For DPO
      kl_coef: 0.1, // For PPO
      alpha: 5.0, // For KTO
    }
  });
  
  useEffect(() => {
    loadModelsAndDatasets();
  }, []);
  
  const loadModelsAndDatasets = async () => {
    setLoading(true);
    try {
      const [modelsResult, datasetsResult] = await Promise.all([
        invoke('get_models'),
        invoke('get_datasets')
      ]);
      
      setModels(modelsResult);
      setDatasets(datasetsResult);
      
      // Set default selections if available
      if (modelsResult.length > 0) {
        setFormData(prev => ({
          ...prev,
          policy_model_id: modelsResult[0].id,
          reference_model_id: modelsResult[0].id,
          reward_model_id: modelsResult.find(m => m.type_name === 'reward')?.id || ''
        }));
      }
      
      if (datasetsResult.length > 0) {
        setFormData(prev => ({
          ...prev,
          dataset_id: datasetsResult[0].id
        }));
      }
      
      setError(null);
    } catch (err) {
      console.error('Failed to load models and datasets:', err);
      setError('Failed to load models and datasets: ' + err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleAlgorithmConfigChange = (name, value) => {
    setFormData(prev => ({
      ...prev,
      algorithm_config: {
        ...prev.algorithm_config,
        [name]: value
      }
    }));
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setCreating(true);
    
    try {
      const result = await invoke('create_experiment', { config: formData });
      onExperimentCreated(result);
    } catch (err) {
      console.error('Failed to create experiment:', err);
      setError('Failed to create experiment: ' + err);
    } finally {
      setCreating(false);
    }
  };
  
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }
  
  return (
    <Box p={2}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>Create New Experiment</Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>
        )}
        
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                label="Experiment Name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                fullWidth
                required
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth required>
                <InputLabel>Algorithm</InputLabel>
                <Select
                  name="algorithm"
                  value={formData.algorithm}
                  onChange={handleChange}
                >
                  <MenuItem value="dpo">Direct Preference Optimization (DPO)</MenuItem>
                  <MenuItem value="ppo">Proximal Policy Optimization (PPO)</MenuItem>
                  <MenuItem value="kto">KL-constrained Preference Optimization (KTO)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                label="Description"
                name="description"
                value={formData.description}
                onChange={handleChange}
                fullWidth
                multiline
                rows={2}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth required>
                <InputLabel>Policy Model</InputLabel>
                <Select
                  name="policy_model_id"
                  value={formData.policy_model_id}
                  onChange={handleChange}
                >
                  {models.map(model => (
                    <MenuItem key={model.id} value={model.id}>
                      {model.name} ({model.parameters.toLocaleString()} params)
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth required>
                <InputLabel>Reference Model</InputLabel>
                <Select
                  name="reference_model_id"
                  value={formData.reference_model_id}
                  onChange={handleChange}
                >
                  {models.map(model => (
                    <MenuItem key={model.id} value={model.id}>
                      {model.name} ({model.parameters.toLocaleString()} params)
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            {formData.algorithm === 'ppo' && (
              <Grid item xs={12} md={6}>
                <FormControl fullWidth required>
                  <InputLabel>Reward Model</InputLabel>
                  <Select
                    name="reward_model_id"
                    value={formData.reward_model_id}
                    onChange={handleChange}
                  >
                    {models.filter(m => m.type_name === 'reward').map(model => (
                      <MenuItem key={model.id} value={model.id}>
                        {model.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            )}
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth required>
                <InputLabel>Dataset</InputLabel>
                <Select
                  name="dataset_id"
                  value={formData.dataset_id}
                  onChange={handleChange}
                >
                  {datasets.map(dataset => (
                    <MenuItem key={dataset.id} value={dataset.id}>
                      {dataset.name} ({dataset.size.toLocaleString()} examples)
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>Training Parameters</Typography>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Typography gutterBottom>Batch Size: {formData.batch_size}</Typography>
              <Slider
                value={formData.batch_size}
                onChange={(e, newValue) => setFormData(prev => ({ ...prev, batch_size: newValue }))}
                step={1}
                marks
                min={1}
                max={16}
                valueLabelDisplay="auto"
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Typography gutterBottom>Learning Rate: {formData.learning_rate}</Typography>
              <Slider
                value={Math.log10(formData.learning_rate) + 6} // Convert to slider range
                onChange={(e, newValue) => {
                  const lr = Math.pow(10, newValue - 6); // Convert back to learning rate
                  setFormData(prev => ({ ...prev, learning_rate: lr }));
                }}
                step={0.5}
                min={-1} // 10^-7
                max={3}   // 10^-3
                marks={[
                  { value: -1, label: '1e-7' },
                  { value: 0, label: '1e-6' },
                  { value: 1, label: '1e-5' },
                  { value: 2, label: '1e-4' },
                  { value: 3, label: '1e-3' },
                ]}
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Typography gutterBottom>Max Steps: {formData.max_steps}</Typography>
              <Slider
                value={formData.max_steps}
                onChange={(e, newValue) => setFormData(prev => ({ ...prev, max_steps: newValue }))}
                step={500}
                min={500}
                max={10000}
                marks={[
                  { value: 500, label: '500' },
                  { value: 2000, label: '2k' },
                  { value: 5000, label: '5k' },
                  { value: 10000, label: '10k' },
                ]}
                valueLabelDisplay="auto"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>Algorithm Configuration</Typography>
            </Grid>
            
            {formData.algorithm === 'dpo' && (
              <Grid item xs={12} md={4}>
                <Typography gutterBottom>Beta: {formData.algorithm_config.beta}</Typography>
                <Slider
                  value={formData.algorithm_config.beta}
                  onChange={(e, newValue) => handleAlgorithmConfigChange('beta', newValue)}
                  step={0.05}
                  min={0.05}
                  max={0.5}
                  marks={[
                    { value: 0.05, label: '0.05' },
                    { value: 0.1, label: '0.1' },
                    { value: 0.2, label: '0.2' },
                    { value: 0.5, label: '0.5' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Grid>
            )}
            
            {formData.algorithm === 'ppo' && (
              <>
                <Grid item xs={12} md={4}>
                  <Typography gutterBottom>KL Coefficient: {formData.algorithm_config.kl_coef}</Typography>
                  <Slider
                    value={formData.algorithm_config.kl_coef}
                    onChange={(e, newValue) => handleAlgorithmConfigChange('kl_coef', newValue)}
                    step={0.05}
                    min={0.01}
                    max={0.5}
                    marks={[
                      { value: 0.01, label: '0.01' },
                      { value: 0.1, label: '0.1' },
                      { value: 0.2, label: '0.2' },
                      { value: 0.5, label: '0.5' },
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Typography gutterBottom>PPO Epochs: {formData.algorithm_config.ppo_epochs || 4}</Typography>
                  <Slider
                    value={formData.algorithm_config.ppo_epochs || 4}
                    onChange={(e, newValue) => handleAlgorithmConfigChange('ppo_epochs', newValue)}
                    step={1}
                    min={1}
                    max={10}
                    marks
                    valueLabelDisplay="auto"
                  />
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Typography gutterBottom>Clip Range: {formData.algorithm_config.clip_range || 0.2}</Typography>
                  <Slider
                    value={formData.algorithm_config.clip_range || 0.2}
                    onChange={(e, newValue) => handleAlgorithmConfigChange('clip_range', newValue)}
                    step={0.05}
                    min={0.1}
                    max={0.5}
                    marks={[
                      { value: 0.1, label: '0.1' },
                      { value: 0.2, label: '0.2' },
                      { value: 0.3, label: '0.3' },
                      { value: 0.5, label: '0.5' },
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Grid>
              </>
            )}
            
            {formData.algorithm === 'kto' && (
              <Grid item xs={12} md={4}>
                <Typography gutterBottom>Alpha: {formData.algorithm_config.alpha}</Typography>
                <Slider
                  value={formData.algorithm_config.alpha}
                  onChange={(e, newValue) => handleAlgorithmConfigChange('alpha', newValue)}
                  step={0.5}
                  min={1}
                  max={10}
                  marks={[
                    { value: 1, label: '1' },
                    { value: 5, label: '5' },
                    { value: 10, label: '10' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Grid>
            )}
            
            <Grid item xs={12} md={4}>
              <TextField
                label="Random Seed"
                name="seed"
                type="number"
                value={formData.seed}
                onChange={handleChange}
                fullWidth
              />
            </Grid>
            
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.use_lora || false}
                    onChange={(e) => setFormData(prev => ({ ...prev, use_lora: e.target.checked }))}
                  />
                }
                label="Use LoRA for efficient fine-tuning"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  disabled={creating}
                  startIcon={creating && <CircularProgress size={20} />}
                >
                  {creating ? 'Creating...' : 'Create Experiment'}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </form>
      </Paper>
    </Box>
  );
};

export default CreateExperiment;