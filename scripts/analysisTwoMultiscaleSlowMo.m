clear all;
close all;
%setenv('GNUTERM','X11')
addpath('../../../mlab/util/');

workspace_path = '../../../';
output_dir = '../outputSlowMoImg/';
checkpoint_dir = '../outputSlowMoImg/Checkpoints/';

max_history = 100000000;
numarbors = 1;
affav1 = [];
av1 = [];

affap1 =[];
ap1 = [];


affap2 =[];
ap2 = [];

analyze_Recon_flag = true;
if analyze_Recon_flag
   Input_list = ...
   {['Input']};
   Recon_list = ...
   {['InputRecon']};
   V1_list = ...
   {['InputReconV1']};
   P1_list = ...
   {['InputReconP1']};
   P2_list = ...
   {['InputReconP2']};
   ffaV1_list = ...
   {['FFAInputReconV1']};
   ffaP1_list = ...
   {['FFAInputReconP1']};
   ffaP2_list = ...
   {['FFAInputReconP2']};
   frameSkip = 1; %Print every 10th frame

   numRecons = size(Input_list, 1);
   assert(numRecons == size(Recon_list, 1));

   outDir = [output_dir, '/Recons/'];
   mkdir(outDir);
   for i_recon = 1:numRecons
      [inputData, inputHdr] = readpvpfile([output_dir, Input_list{i_recon}, '.pvp']);
      [reconData, reconHdr] = readpvpfile([output_dir, Recon_list{i_recon}, '.pvp']);
      [v1Data, v1Hdr] = readpvpfile([output_dir, V1_list{i_recon}, '.pvp']);
      [p1Data, p1Hdr] = readpvpfile([output_dir, P1_list{i_recon}, '.pvp']);
      [p2Data, p2Hdr] = readpvpfile([output_dir, P2_list{i_recon}, '.pvp']);
      
      [ffav1Data, ffav1Hdr] = readpvpfile([output_dir, ffaV1_list{i_recon}, '.pvp']);
      [ffap1Data, ffap1Hdr] = readpvpfile([output_dir, ffaP1_list{i_recon}, '.pvp']);
      [ffap2Data, ffap2Hdr] = readpvpfile([output_dir, ffaP2_list{i_recon}, '.pvp']);
      numFrames = min(length(inputData), length(reconData));
      for i_frame = 1:frameSkip:numFrames
         %readpvpfile returns in [x, y, f]. Octave expects [y, x, f]
         inputImage = permute(inputData{i_frame}.values, [2, 1, 3]);
         reconImage = permute(reconData{i_frame}.values, [2, 1, 3]);
         v1Image = permute(v1Data{i_frame}.values, [2, 1, 3]);
         p1Image = permute(p1Data{i_frame}.values, [2, 1, 3]);
         p2Image = permute(p2Data{i_frame}.values, [2, 1, 3]);
         ffav1Image = permute(ffav1Data{i_frame}.values, [2, 1, 3]);
         ffap1Image = permute(ffap1Data{i_frame}.values, [2, 1, 3]);
         ffap2Image = permute(ffap2Data{i_frame}.values, [2, 1, 3]);
         
         reconObj = v1Image + p1Image + p2Image;
         reconFace = ffav1Image + ffap1Image + ffap2Image;
         
         mixv1 = v1Image + ffav1Image;
         mixp1 = p1Image + ffap1Image;
         mixp2 = p2Image + ffap2Image;
         
         mixv1p1 = mixv1 + mixp1;
         mixp1p2 = mixp1 + mixp2;
         
         time = inputData{i_frame}.time;
         batch = mod(i_frame-1, inputHdr.nbatch);
         assert(time == reconData{i_frame}.time);
         
         maxim = [max(inputImage(:)), max(reconImage(:)), max(v1Image(:)), max(p1Image(:)), max(p2Image(:)), max(ffav1Image(:)), max(ffap1Image(:)), max(ffap2Image(:)), max(reconObj(:)), max(reconFace(:))] ;
         diffmax = max(maxim) - maxim;
         %Normalize
         scaled_inputImage = (inputImage - min(inputImage(:)))/(max(inputImage(:))-min(inputImage(:)));
         scaled_reconImage = (reconImage - min(reconImage(:)))/(max(reconImage(:))-min(reconImage(:)));
         scaled_v1Image = ((v1Image - min(v1Image(:))))/(max(maxim)-min(v1Image(:)));
         scaled_p1Image = ((p1Image - min(p1Image(:))))/(max(maxim)-min(p1Image(:)));
         scaled_p2Image = ((p2Image - min(p2Image(:))))/(max(maxim)-min(p2Image(:)));
         scaled_ffav1Image = ((ffav1Image - min(ffav1Image(:))))/(max(maxim)-min(ffav1Image(:)));
         scaled_ffap1Image = ((ffap1Image - min(ffap1Image(:))))/(max(maxim)-min(ffap1Image(:)));
         scaled_ffap2Image = ((ffap2Image - min(ffap2Image(:))))/(max(maxim)-min(ffap2Image(:)));
         
         scaled_reconObj = ((reconObj - min(reconObj(:))))/(max(maxim)-min(reconObj(:)));
         scaled_reconFace = ((reconFace - min(reconFace(:))))/(max(maxim)-min(reconFace(:)));
         
         scaled_mixv1 = (mixv1 - min(mixv1(:)))/(max(maxim)-min(mixv1(:)));
         scaled_mixp1 = (mixp1 - min(mixp1(:)))/(max(maxim)-min(mixp1(:)));
         scaled_mixp2 = (mixp2 - min(mixp2(:)))/(max(maxim)-min(mixp2(:)));
         scaled_mixv1p1 = (mixv1p1 - min(mixv1p1(:)))/(max(mixv1p1(:))-min(mixv1p1(:)));
         scaled_mixp1p2 = (mixp1p2 - min(mixp1p2(:)))/(max(mixp1p2(:))-min(mixp1p2(:)));
         %scaled_inputImage = (inputImage - min(inputImage(:)))/(max(inputImage(:))-min(inputImage(:)));
         %scaled_reconImage = (reconImage - min(reconImage(:)))/(max(reconImage(:))-min(reconImage(:)));
         %scaled_v1Image = (v1Image - min(v1Image(:)))/(max(v1Image(:))-min(v1Image(:)));
         %scaled_p1Image = (p1Image - min(p1Image(:)))/(max(p1Image(:))-min(p1Image(:)));
         %scaled_p2Image = (p2Image - min(p2Image(:)))/(max(p2Image(:))-min(p2Image(:)));
         %scaled_ffav1Image = (ffav1Image - min(ffav1Image(:)))/(max(ffav1Image(:))-min(ffav1Image(:)));
         %scaled_ffap1Image = (ffap1Image - min(ffap1Image(:)))/(max(ffap1Image(:))-min(ffap1Image(:)));
         %scaled_ffap2Image = (ffap2Image - min(ffap2Image(:)))/(max(ffap2Image(:))-min(ffap2Image(:)));
         %Concat images
         outImg = [scaled_reconImage; scaled_mixv1; scaled_mixp1; scaled_mixp2];
         %Write image
         outName = sprintf('%s/recon_%06d_%03d_%03d.png', outDir, time, batch,i_frame)
         imwrite(outImg, outName);
      end
   end
end



analyze_Recon_flag = true;
if analyze_Recon_flag
   Input_list = ...
   {['Input']};
   Recon_list = ...
   {['InputRecon']};
   V1_list = ...
   {['V1']};
   P1_list = ...
   {['P1']};
   P2_list = ...
   {['P2']};
   ffaV1_list = ...
   {['FFAV1']};
   ffaP1_list = ...
   {['FFAP1']};
   ffaP2_list = ...
   {['FFAP2']};
   frameSkip = 1; %Print every 10th frame

   numRecons = size(Input_list, 1);
   assert(numRecons == size(Recon_list, 1));

   outDir = [output_dir, '/Recons/'];
   mkdir(outDir);
   for i_recon = 1:numRecons
      [inputData, inputHdr] = readpvpfile([output_dir, Input_list{i_recon}, '.pvp']);
      [reconData, reconHdr] = readpvpfile([output_dir, Recon_list{i_recon}, '.pvp']);
      [v1Data, v1Hdr] = readpvpfile([output_dir, V1_list{i_recon}, '.pvp']);
      [p1Data, p1Hdr] = readpvpfile([output_dir, P1_list{i_recon}, '.pvp']);
      [p2Data, p2Hdr] = readpvpfile([output_dir, P2_list{i_recon}, '.pvp']);
      
      [ffav1Data, ffav1Hdr] = readpvpfile([output_dir, ffaV1_list{i_recon}, '.pvp']);
      [ffap1Data, ffap1Hdr] = readpvpfile([output_dir, ffaP1_list{i_recon}, '.pvp']);
      [ffap2Data, ffap2Hdr] = readpvpfile([output_dir, ffaP2_list{i_recon}, '.pvp']);
      numFrames = min(length(inputData), length(reconData));
      for i_frame = 1:frameSkip:numFrames
          if size(v1Data{i_frame,1}.values) > 0
            av1(i_frame) = mean(abs(v1Data{i_frame,1}.values(:,2)));
          else
              av1(i_frame) = 0;
          end
          if size(ffav1Data{i_frame,1}.values) > 0
            affav1(i_frame) = mean(abs(ffav1Data{i_frame,1}.values(:,2)));
          else
            affav1(i_frame) = 0;
          end
          
          if size(p1Data{i_frame,1}.values) > 0
            ap1(i_frame) = mean(abs(p1Data{i_frame,1}.values(:,2)));
          else
              ap1(i_frame) = 0;
          end
          if size(ffap1Data{i_frame,1}.values) > 0
            affap1(i_frame) = mean(abs(ffap1Data{i_frame,1}.values(:,2)));
          else
            affap1(i_frame) = 0;
          end
          
          if size(p2Data{i_frame,1}.values) > 0
            ap2(i_frame) = mean(abs(p2Data{i_frame,1}.values(:,2)));
          else
              ap2(i_frame) = 0;
          end
          if size(ffap2Data{i_frame,1}.values) > 0
            affap2(i_frame) = mean(abs(ffap2Data{i_frame,1}.values(:,2)));
          else
            affap2(i_frame) = 0;
          end
      end
   end
end

plot_flag = 1;
analyze_Sparse_flag = true;
if analyze_Sparse_flag
    Sparse_list = ...
       {[''], ['FFAP2']; [''],['P2']; [''],['FFAP1']; [''],['P1']; [''],['FFAV1']; [''],['V1']; ...
        };

    load_Sparse_flag = 0;
    plot_flag = 0;

    fraction_Sparse_frames_read = 1;
    min_Sparse_skip = 1;
    fraction_Sparse_progress = 1;
    num_procs = 8;
    num_epochs = 1;
    Sparse_frames_list = [];

  [Sparse_hdr, ...
   Sparse_hist_rank_array, ...
   Sparse_times_array, ...
   Sparse_percent_active_array, ...
   Sparse_percent_change_array, ...
   Sparse_std_array, ...
   Sparse_struct_array] = ...
      analyzeSparseEpochsPVP2(Sparse_list, ...
			     output_dir, ...
			     load_Sparse_flag, ...
			     plot_flag, ...
			     fraction_Sparse_frames_read, ...
			     min_Sparse_skip, ...
			     fraction_Sparse_progress, ...
			     Sparse_frames_list, ...
			     num_procs, ...
			     num_epochs);
end


figure
plot(affap2, 'LineWidth', 2);
hold on;
plot(ap2, 'LineWidth', 2);
plot(affap1, 'LineWidth', 2);
plot(ap1, 'LineWidth', 2);
plot(affav1, 'LineWidth', 2);
plot(av1, 'LineWidth', 2);
set(gcf,'color','w');
set(gca,'FontSize',18)
legend('Face FFA','Obj IT', 'Face V2', 'Obj V2', 'Face V1', 'Obj V1');
xlabel('Timestep') ;
ylabel('Magnitude of Response') ;
hold off;

figure;
plot(Sparse_percent_active_array{1}*100, 'LineWidth', 2);
hold on;
plot(Sparse_percent_active_array{2}*100, 'LineWidth', 2);
plot(Sparse_percent_active_array{3}*100, 'LineWidth', 2);
plot(Sparse_percent_active_array{4}*100, 'LineWidth', 2);
plot(Sparse_percent_active_array{5}*100, 'LineWidth', 2);
plot(Sparse_percent_active_array{6}*100, 'LineWidth', 2);
legend('Face FFA','Obj IT', 'Face V2', 'Obj V2', 'Face V1', 'Obj V1');
set(gcf,'color','w');
set(gca,'FontSize',18)
xlabel('Timestep') ;
ylabel('Percent Neurons Active') ;
analyze_nonSparse_flag = true;
if analyze_nonSparse_flag
    nonSparse_list = ...
        {[''], ['InputTextError']; ...
         };
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(10, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {...
         [''], ['InputText']; ...
         }; ...
    nonSparse_norm_strength = [1 1];
    Sparse_std_ndx = [0 0];
    plot_flag = true;

  if ~exist('Sparse_std_ndx')
    Sparse_std_ndx = zeros(num_nonSparse_list,1);
  end
  if ~exist('nonSparse_norm_strength')
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
  end

  fraction_nonSparse_frames_read = 1;
  min_nonSparse_skip = 1;
  fraction_nonSparse_progress = 10;
  [nonSparse_times_array, ...
   nonSparse_RMS_array, ...
   nonSparse_norm_RMS_array, ...
   nonSparse_RMS_fig] = ...
      analyzeNonSparsePVP(nonSparse_list, ...
		       nonSparse_skip, ...
		       nonSparse_norm_list, ...
		       nonSparse_norm_strength, ...
		       Sparse_times_array, ...
		       Sparse_std_array, ...
		       Sparse_std_ndx, ...
		       output_dir, ...
		       plot_flag, ...
		       fraction_nonSparse_frames_read, ...
		       min_nonSparse_skip, ...
		       fraction_nonSparse_progress);

end %% analyze_nonSparse_flag

%%keyboard;
plot_flag = false;
plot_weights = true;
if plot_weights
   weights_list = ...
   { ...
   ['V1ToInputError_W']; ...
   };
   pre_list = ...
   { ...
   ['V1_A']; ...
   };
   sparse_ndx = ...
   [   ...
   1;  ...
   ];

   checkpoints_list = {dir(checkpoint_dir).name};
   %Remove hidden files
   for i = length(checkpoints_list):-1:1
      % remove folders starting with .
      fname = checkpoints_list{i};
      if fname(1) == '.'
         checkpoints_list(i) = [ ];
      end
   end

   num_checkpoints = length(checkpoints_list);
   checkpoint_weights_movie = true;
   no_clobber = false;
   weights_movie_dir = [output_dir, filesep, 'weights_movie']

   num_weights_list = size(weights_list,1);
   weights_hdr = cell(num_weights_list,1);
   pre_hdr = cell(num_weights_list,1);
   if checkpoint_weights_movie
      weights_movie_dir = [output_dir, filesep, 'weights_movie']
      [status, msg, msgid] = mkdir(weights_movie_dir);
      if status ~= 1
         warning(['mkdir(', weights_movie_dir, ')', ' msg = ', msg]);
      end 
   end
   if(plot_flag)
      weights_dir = [output_dir, filesep, 'weights']
      [status, msg, msgid] = mkdir(weights_dir);
      if status ~= 1
         warning(['mkdir(', weights_dir, ')', ' msg = ', msg]);
      end 
   end
   for i_weights = 1 : num_weights_list
      max_weight_time = 0;
      max_checkpoint = 0;
      for i_checkpoint = 1 : num_checkpoints
         checkpoint_path = [checkpoint_dir, checkpoints_list{i_checkpoint}];
         weights_file = [checkpoint_path, filesep, weights_list{i_weights,1}, '.pvp'];
         if ~exist(weights_file, 'file')
            warning(['file does not exist: ', weights_file]);
            continue;
         end
         weights_fid = fopen(weights_file);
         weights_hdr{i_weights} = readpvpheader(weights_fid);    
         fclose(weights_fid);

         weight_time = weights_hdr{i_weights}.time;
         if weight_time > max_weight_time
              max_weight_time = weight_time;
              max_checkpoint = i_checkpoint;
         end
      end %% i_checkpoint

      for i_checkpoint = 1 : num_checkpoints
         checkpoint_path = [checkpoint_dir, checkpoints_list{i_checkpoint}];
         weights_file = [checkpoint_path, filesep, weights_list{i_weights,1}, '.pvp'];
         if ~exist(weights_file, 'file')
            warning(['file does not exist: ', weights_file]);
            continue;
         end
         weights_fid = fopen(weights_file);
         weights_hdr{i_weights} = readpvpheader(weights_fid);    
         fclose(weights_fid);
         weights_filedata = dir(weights_file);
         patchsize = weights_hdr{i_weights}.nxp*weights_hdr{i_weights}.nyp*weights_hdr{i_weights}.nfp;
         numpatches = weights_hdr{i_weights}.numPatches;
         datasize = weights_hdr{i_weights}.datasize;
         weights_framesize = (patchsize * datasize + 8) * numpatches + weights_hdr{i_weights}.headersize;
         tot_weights_frames = weights_filedata(1).bytes/weights_framesize;
         num_weights = 1;
         progress_step = ceil(tot_weights_frames / 10);
         [weights_struct, weights_hdr_tmp] = ...
         readpvpfile(weights_file, progress_step, tot_weights_frames, tot_weights_frames-num_weights+1);
         i_frame = num_weights;
         i_arbor = 1;
         weight_vals = squeeze(weights_struct{i_frame}.values{i_arbor});
         weight_time = squeeze(weights_struct{i_frame}.time);
         weights_name =  [weights_list{i_weights,1}, '_', num2str(weight_time, '%08d')];
         if no_clobber && exist([weights_movie_dir, filesep, weights_name, '.png']) && i_checkpoint ~= max_checkpoint
            continue;
         end
         tmp_ndx = sparse_ndx(i_weights);
         if analyze_Sparse_flag
            tmp_rank = Sparse_hist_rank_array{tmp_ndx};
         else
            tmp_rank = [];
         end
         if analyze_Sparse_flag && ~isempty(tmp_rank)
            pre_hist_rank = tmp_rank;
         else
             pre_hist_rank = (1:weights_hdr{i_weights}.nf);
         end
         %% make tableau of all patches
         %%keyboard;
         i_patch = 1;
         num_weights_dims = ndims(weight_vals);
         num_patches = size(weight_vals, num_weights_dims);
         num_patches_rows = floor(sqrt(num_patches));
         num_patches_cols = ceil(num_patches / num_patches_rows);
         num_weights_colors = 1;
         if num_weights_dims == 4
             num_weights_colors = size(weight_vals,3);
         end
         if plot_flag && i_checkpoint == max_checkpoint
            weights_fig = figure;
            set(weights_fig, 'name', weights_name);
         end
         weight_patch_array = [];
         for j_patch = 1  : num_patches
            i_patch = pre_hist_rank(j_patch);
            if plot_flag && i_checkpoint == max_checkpoint
               subplot(num_patches_rows, num_patches_cols, j_patch); 
            end
            if num_weights_colors == 1
               patch_tmp = squeeze(weight_vals(:,:,i_patch));
            else
               patch_tmp = squeeze(weight_vals(:,:,:,i_patch));
            end
            patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
            min_patch = min(patch_tmp2(:));
            max_patch = max(patch_tmp2(:));
            patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch - min_patch)==0));
            patch_tmp2 = uint8(permute(patch_tmp2, [2,1,3])); %% uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));
            if plot_flag && i_checkpoint == max_checkpoint
               imagesc(patch_tmp2); 
               if num_weights_colors == 1
                  colormap(gray);
               end
               box off
               axis off
               axis image
            end %% plot_flag
            if isempty(weight_patch_array)
               weight_patch_array = ...
               zeros(num_patches_rows*size(patch_tmp2,1), num_patches_cols*size(patch_tmp2,2), size(patch_tmp2,3));
            end
            col_ndx = 1 + mod(j_patch-1, num_patches_cols);
            row_ndx = 1 + floor((j_patch-1) / num_patches_cols);
            weight_patch_array(((row_ndx-1)*size(patch_tmp2,1)+1):row_ndx*size(patch_tmp2,1), ...
            ((col_ndx-1)*size(patch_tmp2,2)+1):col_ndx*size(patch_tmp2,2),:) = ...
            patch_tmp2;
         end  %% j_patch
         if plot_flag && i_checkpoint == max_checkpoint
            saveas(weights_fig, [weights_dir, filesep, weights_name, '.png'], 'png');
         end
         imwrite(uint8(weight_patch_array), [weights_movie_dir, filesep, weights_name, '.png'], 'png');
         %% make histogram of all weights
         if plot_flag && i_checkpoint == max_checkpoint
            weights_hist_fig = figure;
            [weights_hist, weights_hist_bins] = hist(weight_vals(:), 100);
            bar(weights_hist_bins, log(weights_hist+1));
            set(weights_hist_fig, 'name', ...
            ['Hist_',  weights_list{i_weights,1}, '_', num2str(weight_time, '%08d')]);
            saveas(weights_hist_fig, ...
            [weights_dir, filesep, 'weights_hist_', num2str(weight_time, '%08d')], 'png');
         end
      end %% i_checkpoint
   end %% i_weights
end  %% plot_weights
disp("Finished\n");
