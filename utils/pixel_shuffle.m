function out = pixel_shuffle(in, scale)
sz = size(in);
out = zeros(sz(1)*scale, sz(2)*scale);
    for c = 1:scale*scale
        q = floor((c-1)/scale)+1;
        r = mod(c, scale);
        if r == 0, r = scale; end
        out(q:scale:end, r:scale:end) = in(:, :, 1, c);
    end
end